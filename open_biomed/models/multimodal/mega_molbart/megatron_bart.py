from megatron.module import MegatronModule
from apex.normalization import FusedLayerNorm
from megatron import mpu
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from functools import partial
from .tokenizer import load_tokenizer
from .util import DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH, REGEX


class MultiheadAttention(MegatronModule):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        cross_attention=False,
        init_method=init.xavier_uniform_,
        ):

        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(p=dropout)
        self.bias = bias
        self.cross_attention = cross_attention
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.init_method = init_method
        self.skip_bias = not bias

        # Self-Attention is Column Parallelized
        self.query_key_value = mpu.ColumnParallelLinear(self.embed_dim,
                3 * self.embed_dim, gather_output=True,
                init_method=self.init_method,
                skip_bias_add=self.skip_bias)

        # Cross-Attention is Row and Column Parallelized
        self.q_proj = mpu.RowParallelLinear(self.embed_dim,
                self.embed_dim, input_is_parallel=False,
                init_method=self.init_method, bias=bias,
                skip_bias_add=self.skip_bias)
        self.key_value = mpu.ColumnParallelLinear(self.embed_dim, 2
                * self.embed_dim, gather_output=True,
                init_method=self.init_method,
                skip_bias_add=self.skip_bias)

        # Final projection is Row Parallelized
        self.out_proj = mpu.RowParallelLinear(self.embed_dim,
                self.embed_dim, input_is_parallel=False,
                init_method=self.init_method, bias=bias)

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        ):
        """Input shape: Time x Batch x Channel

        Args:
            query - tokens/states of shape [Time x Batch x Channel]
            key - tokens/states of shape [Time x Batch x Channel]
            value - tokens/states of shape [Time x Batch x Channel]
            key_padding_mask - keys that are pads where padding
                elements are indicated by 1s. Shape: [batch, src_len].
            attn_mask - typically used to implement causal attention, where
                the mask prevents the attention from looking forward in time.
                Shape: [tgt_len, src_len].
        Returns:
            outputs - attention probability scores of shape (Time x Batch x Channel)
        """

        (tgt_len, bsz, embed_dim) = query.size()

        # Compute attention projections
        if not self.cross_attention:
            (q_k_v, bias) = self.query_key_value(query)
            (q, k, v) = mpu.split_tensor_along_last_dim(q_k_v, 3)
        else:
            q, _ = self.q_proj(query)
            if key is None:
                assert value is None, \
                    'Cross attention mode: since key is None, value must also be None.'
                k = v = None
            else:
                (k_v, bias) = self.key_value(key)
                (k, v) = mpu.split_tensor_along_last_dim(k_v, 2)

        # Scale query and reshape
        q = q.contiguous()
        q *= self.scaling
        q = q.view(tgt_len, bsz * self.num_heads,
                   self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)

        # Compute attention scores
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads,
                tgt_len, src_len]

        # Apply causal attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        # Apply padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads,
                    tgt_len, src_len)
            attn_weights = \
                attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads,
                    tgt_len, src_len)

        # Compute attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)

        # Compute context and output projection
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len,
                self.head_dim]
        if attn.size(1) == 1:  # a single decoder step (sequence length == 1)
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
                    embed_dim)
        (attn, bias) = self.out_proj(attn)
        attn_output_weights = attn_probs.view(bsz, self.num_heads,
                tgt_len, src_len)
        attn_output_weights = attn_output_weights.sum(dim=1) \
            / self.num_heads
        return (attn, attn_output_weights)


class EncoderLayer(MegatronModule):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            cross_attention=False,
            init_method=init_method,
            )
        self.self_attn_layer_norm = FusedLayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.gelu
        self.activation_dropout = nn.Dropout(p=dropout)
        self.fc1 = mpu.ColumnParallelLinear(embed_dim, 4
                * embed_dim, gather_output=False,
                init_method=init_method, skip_bias_add=False)
        self.fc2 = mpu.RowParallelLinear(4 * embed_dim,
                embed_dim, input_is_parallel=True,
                init_method=init_method, skip_bias_add=False)
        self.final_layer_norm = FusedLayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_padding_mask=None,
        attn_mask=None,
        ):
        """
        Args:
            x: input to the layer of shape (seq_len, batch, embed_dim)
            encoder_padding_mask: binary ByteTensor of shape
                (batch, seq_len) where padding elements are indicated by 1.
            attn_mask: binary tensor of shape (tgt_len, src_len),
                where tgt_len is the length of output and src_len is the
                length of input, though here both are equal to seq_len.
        Returns:
            encoded output of shape (seq_len, batch, embed_dim)
        """

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool),
                    -1e8)
        residual = x
        x = self.self_attn_layer_norm(x)
        (x, weights) = self.self_attn(query=x, key=x, value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask)
        x = self.attn_dropout(x)
        x = x + residual
        residual = x
        x = self.final_layer_norm(x)
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x, _ = self.fc2(x)
        x = self.attn_dropout(x)
        x = x + residual
        return x


class DecoderLayer(MegatronModule):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            cross_attention=False,
            init_method=init_method,
            )
        self.self_attn_layer_norm = FusedLayerNorm(embed_dim)
        self.encoder_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            cross_attention=True,
            init_method=init_method,
            )
        self.encoder_attn_layer_norm = FusedLayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.gelu
        self.activation_dropout = nn.Dropout(p=dropout)
        self.fc1 = mpu.ColumnParallelLinear(embed_dim, 4
                * embed_dim, gather_output=False,
                init_method=init_method, skip_bias_add=False)
        self.fc2 = mpu.RowParallelLinear(4 * embed_dim,
                embed_dim, input_is_parallel=True,
                init_method=init_method, skip_bias_add=False)
        self.final_layer_norm = FusedLayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        ):
        """
        Args:
            x: input to decoder layer of shape (seq_len, batch, embed_dim)
            encoder_out: output from the encoder
            encoder_padding_mask: binary ByteTensor of shape
                (batch, seq_len) where padding elements are indicated by 1
            self_attn_mask: binary tensor of shape (tgt_len, src_len),
                where tgt_lent is the length of output and src_len is the
                length of input, though here both are equal to seq_len.
            self_attn_padding_mask: binary ByteTensor of shape
                (batch, seq_len) where padding elements are indicated by 1.
        Returns:
            encoded output of shape (seq_len, batch, embed_dim)
        """

        residual = x
        x = self.self_attn_layer_norm(x)

        # Self-Attention block

        (x, weights) = self.self_attn(query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask)
        x = self.dropout(x)
        x = x + residual

        # Cross-Attention block
        if encoder_out is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            (x, attn) = self.encoder_attn(query=x, key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask)
            x = self.dropout(x)
            x = x + residual
        residual = x
        x = self.final_layer_norm(x)

        # Fully-connected block
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x, _ = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        return x


class ParallelTransformerEncoder(MegatronModule):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(ParallelTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = dropout
        self.bias = bias
        self.init_method = init_method
        self.layers.extend([self.build_encoder_layer() for i in
                           range(self.num_layers)])
        self.norm = FusedLayerNorm(self.embed_dim)

    def build_encoder_layer(self):
        layer = EncoderLayer(self.embed_dim, self.num_heads,
                             dropout=self.attn_dropout, bias=self.bias,
                             init_method=self.init_method)
        return layer

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        ):
        """Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Returns:
            encoded output of shape (src_len, batch, embed_dim)
        """

        output = src
        for mod in self.layers:
            output = mod(output, attn_mask=mask,
                         encoder_padding_mask=src_key_padding_mask)
        output = self.norm(output)
        return output


class ParallelTransformerDecoder(MegatronModule):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(ParallelTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = dropout
        self.bias = bias
        self.init_method = init_method
        self.layers.extend([self.build_decoder_layer() for i in
                           range(self.num_layers)])
        self.norm = FusedLayerNorm(self.embed_dim)

    def build_decoder_layer(self):
        layer = DecoderLayer(self.embed_dim, self.num_heads,
                             dropout=self.attn_dropout, bias=self.bias,
                             init_method=self.init_method)
        return layer

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        ):
        """Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Returns:
            decoded output of shape (tgt_len, batch, embed_dim)
        """

        output = tgt
        for mod in self.layers:
            output = mod(output, encoder_out=memory,
                         encoder_padding_mask=memory_key_padding_mask,
                         self_attn_mask=tgt_mask,
                         self_attn_padding_mask=tgt_key_padding_mask)
        output = self.norm(output)
        return output


class MegatronBART(MegatronModule):

    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        max_seq_len,
        dropout=0.0,
        ):

        super().__init__()

        self.sampler = decode_sampler
        self.pad_token_idx = pad_token_idx
        self.val_sampling_alg = 'greedy'
        self.num_beams = 5
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.emb_dropout = nn.Dropout(p=dropout)
        init_method = init.xavier_uniform_

        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = dropout
        self.encoder = ParallelTransformerEncoder(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dropout,
            bias=True,
            init_method=init_method,
            )
        self.decoder = ParallelTransformerDecoder(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dropout,
            bias=True,
            init_method=init_method,
            )
        self.token_fc = mpu.RowParallelLinear(d_model, vocab_size,
                input_is_parallel=False, init_method=init_method,
                skip_bias_add=False)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none',
                ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self._init_params(init_method)
        self.register_buffer('pos_emb', self._positional_embs())

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x['encoder_input']
        decoder_input = x['decoder_input']
        encoder_pad_mask = x['encoder_pad_mask'].transpose(0, 1)
        decoder_pad_mask = x['decoder_pad_mask'].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)

        (seq_len, _, _) = tuple(decoder_embs.size())
        tgt_mask = \
            self._generate_square_subsequent_mask(seq_len).to(decoder_embs.device)

        memory = self.encoder(encoder_embs,
                              src_key_padding_mask=encoder_pad_mask)
        model_output = self.decoder(decoder_embs, memory,
                                    tgt_mask=tgt_mask,
                                    tgt_key_padding_mask=decoder_pad_mask,
                                    memory_key_padding_mask=encoder_pad_mask.clone())

        token_output, _ = self.token_fc(model_output)
        output = {'model_output': model_output,
                  'token_output': token_output}

        return output

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch['encoder_input']
        encoder_pad_mask = batch['encoder_pad_mask'].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs,
                                    src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch['decoder_input']
        decoder_pad_mask = batch['decoder_pad_mask'].transpose(0, 1)
        memory_input = batch['memory_input']
        memory_pad_mask = batch['memory_pad_mask'].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        (seq_len, _, _) = tuple(decoder_embs.size())
        tgt_mask = \
            self._generate_square_subsequent_mask(seq_len).to(decoder_embs.device)

        model_output = self.decoder(decoder_embs, memory_input,
                                    tgt_key_padding_mask=decoder_pad_mask,
                                    memory_key_padding_mask=memory_pad_mask,
                                    tgt_mask=tgt_mask)
        token_output, _  = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def validation_step(self, batch, batch_idx=None):
        self.eval()
        # TODO: This can be further optimized
        tokenizer = load_tokenizer(vocab_path=DEFAULT_VOCAB_PATH, chem_token_start=DEFAULT_CHEM_TOKEN_START, regex=REGEX)

        with torch.no_grad():
            model_output = self.forward(batch)
            #target_smiles = batch['target_smiles']
            token_ids = batch['target']
            tokens = token_ids.transpose(0, 1).tolist()
            tokens = tokenizer.convert_ids_to_tokens(tokens)
            target_smiles = tokenizer.detokenize(tokens)

            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_char_acc(batch, model_output)
            perplexity = self._calc_perplexity(batch, model_output)
            (mol_strs, log_lhs) = self.sample_molecules(batch,
                    sampling_alg=self.val_sampling_alg)
            metrics = self.sampler.calc_sampling_metrics(mol_strs,
                    target_smiles)

        self.train()

        val_outputs = {
            'val_loss': loss.item(),
            'val_token_acc': token_acc,
            'val_perplexity': perplexity,
            'val_molecular_accuracy': metrics['accuracy'],
            'val_invalid_smiles': metrics['invalid'],
            }
        return val_outputs

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input['target']
        pad_mask = batch_input['target_pad_mask']
        token_output = model_output['token_output']
        token_mask_loss = self._calc_mask_loss(token_output, tokens,
                pad_mask)
        return token_mask_loss

    def _calc_mask_loss(
        self,
        token_output,
        target,
        target_mask,
        ):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokenizer
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        (seq_len, batch_size) = tuple(target.size())
        token_pred = token_output.reshape((seq_len * batch_size,
                -1)).float()
        loss = self.loss_fn(token_pred,
                            target.reshape(-1)).reshape((seq_len,
                batch_size))
        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens
        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input['target']
        target_mask = batch_input['target_pad_mask']
        vocab_dist_output = model_output['token_output']
        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2,
                target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)
        seq_lengths = inv_target_mask.sum(dim=0)
        exp = -(1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean().item()

    def _calc_char_acc(self, batch_input, model_output):
        token_ids = batch_input['target']
        target_mask = batch_input['target_pad_mask']
        token_output = model_output['token_output']
        target_mask = ~(target_mask > 0)
        (_, pred_ids) = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask
        num_correct = correct_ids.sum()
        total = target_mask.sum()
        accuracy = num_correct / total
        return accuracy

    def sample_molecules(self, batch_input, sampling_alg='greedy'):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input['encoder_input']
        enc_mask = batch_input['encoder_pad_mask']

        # Freezing the weights reduces the amount of memory leakage in the transformer
        #model.eval()

        with torch.no_grad():

            encode_input = {'encoder_input': enc_input,
                            'encoder_pad_mask': enc_mask}
            memory = self.encode(encode_input)
            mem_mask = enc_mask.clone()
            (_, batch_size, _) = tuple(memory.size())
            decode_fn = partial(self._decode_fn, memory=memory,
                                mem_pad_mask=mem_mask)
            #self.sampler.device = self.device
            if sampling_alg == 'greedy':
                (mol_strs, log_lhs) = \
                    self.sampler.greedy_decode(decode_fn, batch_size,device=memory.device)
            elif sampling_alg == 'beam':
                (mol_strs, log_lhs) = \
                    self.sampler.beam_decode(decode_fn, batch_size,
                        self.num_beams,device=memory.device)

        # Must remember to unfreeze!
        #model.train()

        return (mol_strs, log_lhs)

    def _decode_fn(
        self,
        token_ids,
        pad_mask,
        memory,
        mem_pad_mask,
        ):
        decode_input = {
            'decoder_input': token_ids,
            'decoder_pad_mask': pad_mask,
            'memory_input': memory,
            'memory_pad_mask': mem_pad_mask,
            }
        model_output = self.decode(decode_input)
        return model_output

    def _construct_input(self, token_ids, sentence_masks=None):
        (seq_len, _) = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)
        positional_embs = self.pos_emb[:seq_len, :
                ].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.emb_dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0,
                            self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs))
                for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model]
                for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz):
        """ 
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'
                )).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self, method):
        """
        Apply initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                method(p)
