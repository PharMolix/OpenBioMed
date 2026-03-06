# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import logging as std_logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from math import inf
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.esm.modeling_esm import *
from tqdm import tqdm
import os
import random

from dataclasses import dataclass, field


from transformers import EsmTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken
from pytorch_lightning.utilities import rank_zero_only

@dataclass
class SelfMixupConfig:
    enable: bool = field(default=False)
    with_original_loss: bool = field(default=False)


@dataclass
class TokenizerConfig:
    vocab_file: str = field(default="airkingbd/dplm2_650m")
    # amino acid tokens (33) + struct tokens (8192) + 4 special struct tokens
    vocab_size: int = field(default=33 + 8192 + 4)


@dataclass
class StructTokenizerConfig:
    enable: bool = field(default=True)
    exp_path: str = field(default="airkingbd/struct_tokenizer")
    
@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""

@dataclass
class CondConfig:
    use_go: bool = True
    go_num: int = 375
    go_drop: float = 0.5
    use_ipr: bool = True
    ipr_num: int = 1154
    ipr_drop: float = 0.5
    use_ec: bool = False
    ec_num: int = 661
    ec_drop: float = 0.0
    use_seq_motif: bool = False
    use_struc_bb: bool = False
    motif_min_len: int = 10
    motif_max_len: int = 30

@dataclass
class LoRAConfig:
    lora: bool = field(
        default=False
    )
    lora_rank: int = field(
        default=16
    )
    lora_dropout: float = field(
        default=0.1
    )
    lora_target_module: str = field(
        default=""
    )
    modules_to_save: str = field(
        default=""
    )

@dataclass
class CodeFPConfig:
    ## DPLM model
    num_diffusion_timesteps: int = field(default=500)
    tokenizer: TokenizerConfig = field(default=TokenizerConfig())
    lora: LoRAConfig = field(default=LoRAConfig())
    net: NetConfig = field(default=NetConfig())
    gradient_ckpt: bool = field(default=False)

    ## multi-modal training
    training_stage: str = field(default="train_from_dplm")
    self_mixup: SelfMixupConfig = field(
        default=SelfMixupConfig()
    )  # training strategy
    single_modality_ratio: float = field(default=0.25)
    folding_loss_ratio: float = field(default=0.25)
    inverse_folding_loss_ratio: float = field(default=0.25)
    joint_loss_ratio: float = field(default=0.25)
    independent_loss_ratio: float = field(default=0.0)

    ## struct tokenizer
    struct_tokenizer: StructTokenizerConfig = field(
        default=StructTokenizerConfig()
    )

    rdm_couple: bool = field(
        default=False
    )
    cond: CondConfig = field(default_factory=CondConfig)

    use_diff_modulation: bool = field(default=False)
    use_func_cross_attn: bool = field(default=False)
    use_diff_ce: bool = field(default=False)
    use_motif_struct_emb: bool = field(default=False)
    use_static_scale: bool = field(default=False)

    use_attention_store: bool = field(default=False)
    use_go_null_token: bool = field(default=False)
    use_motif_head: bool = field(default=False)

    use_only_struct: bool = field(default=False)

def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]

def get_logger(name=__name__) -> std_logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = std_logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

log = get_logger(__name__)

class DPLM2Tokenizer(EsmTokenizer):
    SPECIAL_TOKENS_ATTRIBUTES = [
        "aa_cls_token",
        "aa_eos_token",
        "aa_unk_token",
        "aa_mask_token",
        "struct_cls_token",
        "struct_eos_token",
        "struct_unk_token",
        "struct_mask_token",
        "pad_token",
    ]

    def __init__(
        self,
        vocab_file,
        aa_cls_token="<cls_aa>",
        aa_eos_token="<eos_aa>",
        aa_unk_token="<unk_aa>",
        aa_mask_token="<mask_aa>",
        struct_cls_token="<cls_struct>",
        struct_eos_token="<eos_struct>",
        struct_unk_token="<unk_struct>",
        struct_mask_token="<mask_struct>",
        pad_token="<pad>",
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {
            tok: ind for ind, tok in enumerate(self.all_tokens)
        }

        self._aa_cls_token = None
        self._aa_eos_token = None
        self._aa_unk_token = None
        self._aa_mask_token = None
        self._struct_cls_token = None
        self._struct_eos_token = None
        self._struct_unk_token = None
        self._struct_mask_token = None
        self._pad_token = None

        PreTrainedTokenizer.__init__(
            self,
            aa_cls_token=aa_cls_token,
            aa_eos_token=aa_eos_token,
            aa_unk_token=aa_unk_token,
            aa_mask_token=aa_mask_token,
            struct_cls_token=struct_cls_token,
            struct_eos_token=struct_eos_token,
            struct_unk_token=struct_unk_token,
            struct_mask_token=struct_mask_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    @property
    def aa_eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_eos_token is None:
            if self.verbose:
                log.error("Using aa_eos_token, but it is not set yet.")
            return None
        return str(self._aa_eos_token)

    @property
    def aa_cls_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_cls_token is None:
            if self.verbose:
                log.error("Using aa_cls_token, but it is not set yet.")
            return None
        return str(self._aa_cls_token)

    @property
    def aa_unk_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_unk_token is None:
            if self.verbose:
                log.error("Using aa_unk_token, but it is not set yet.")
            return None
        return str(self._aa_unk_token)

    @property
    def aa_mask_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_mask_token is None:
            if self.verbose:
                log.error("Using aa_mask_token, but it is not set yet.")
            return None
        return str(self._aa_mask_token)

    @property
    def struct_eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_eos_token is None:
            if self.verbose:
                log.error("Using struct_eos_token, but it is not set yet.")
            return None
        return str(self._struct_eos_token)

    @property
    def struct_cls_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_cls_token is None:
            if self.verbose:
                log.error("Using struct_cls_token, but it is not set yet.")
            return None
        return str(self._struct_cls_token)

    @property
    def struct_unk_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_unk_token is None:
            if self.verbose:
                log.error("Using struct_unk_token, but it is not set yet.")
            return None
        return str(self._struct_unk_token)

    @property
    def struct_mask_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_mask_token is None:
            if self.verbose:
                log.error("Using struct_mask_token, but it is not set yet.")
            return None
        return str(self._struct_mask_token)

    @aa_cls_token.setter
    def aa_cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_cls_token"
            )
        self._aa_cls_token = value

    @aa_eos_token.setter
    def aa_eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_eos_token"
            )
        self._aa_eos_token = value

    @aa_unk_token.setter
    def aa_unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_unk_token"
            )
        self._aa_unk_token = value

    @aa_mask_token.setter
    def aa_mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_mask_token"
            )
        self._aa_mask_token = value

    @struct_cls_token.setter
    def struct_cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_cls_token"
            )
        self._struct_cls_token = value

    @struct_eos_token.setter
    def struct_eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_eos_token"
            )
        self._struct_eos_token = value

    @struct_unk_token.setter
    def struct_unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_unk_token"
            )
        self._struct_unk_token = value

    @struct_mask_token.setter
    def struct_mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_mask_token"
            )
        self._struct_mask_token = value


def _init_module_weights(module, initializer_range=0.02):
    """
    一个辅助函数，用于对单个模块（如 nn.Linear）进行初始化。
    """
    if isinstance(module, nn.Linear):
        # 线性层初始化：使用截断正态分布
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
            
    elif isinstance(module, nn.Embedding):
        # 嵌入层初始化：使用正态分布
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm 初始化
        if hasattr(module.weight, 'data'):
            module.weight.data.fill_(1.0)
        if hasattr(module.bias, 'data'):
            module.bias.data.zero_()


class ModifiedRotaryEmbedding(RotaryEmbedding):
    """Rotary position embeddings based on those in.

    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.aa_type = 1
        self.struct_type = 0

    def _update_cos_sin_tables(self, x, type_ids, seq_dimension=2):
        seq_len = x.shape[seq_dimension]
        if self.aa_type in type_ids and self.struct_type in type_ids:
            seq_len /= 2

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, type_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, type_ids=type_ids, seq_dimension=-2
        )

        if self.aa_type in type_ids and self.struct_type in type_ids:
            q_1, q_2 = q.chunk(2, dim=-2)
            k_1, k_2 = k.chunk(2, dim=-2)
            q_1 = apply_rotary_pos_emb(q_1, self._cos_cached, self._sin_cached)
            q_2 = apply_rotary_pos_emb(q_2, self._cos_cached, self._sin_cached)
            k_1 = apply_rotary_pos_emb(k_1, self._cos_cached, self._sin_cached)
            k_2 = apply_rotary_pos_emb(k_2, self._cos_cached, self._sin_cached)
            q = torch.cat((q_1, q_2), dim=-2)
            k = torch.cat((k_1, k_2), dim=-2)
            return (q, k)
        else:
            return (
                apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
                apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
            )


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.rotary_embeddings = ModifiedRotaryEmbedding(
            dim=self.attention_head_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states)
            )
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states)
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(
                query_layer, key_layer, type_ids
            )

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            raise NotImplementedError

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()
        if hasattr(F, "scaled_dot_product_attention"):
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                scale=1.0,
            )
        else:
            # raise NotImplementedError
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = torch.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_probs, value_layer)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_diff(x, shift, scale):
    b, n, hid = x.shape

    n_half = n // 2

    mask = torch.zeros_like(x)
    mask[:, :n_half, :] = 1

    modulated = (
        (x * (1 + scale[0].unsqueeze(1)) + shift[0].unsqueeze(1))
    ) * mask + (
        (x * (1 + scale[1].unsqueeze(1)) + shift[1].unsqueeze(1))
    ) * (1 - mask)

    return modulated


class AGFMSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, gate=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if gate is not None:
            if isinstance(gate, list):
                gate1, gate2 = gate
                n = hidden_states.shape[1]
                n_half = n // 2
                    
                mask = torch.zeros_like(hidden_states)
                mask[:, :n_half, :] = 1

                modulated_part1 = gate1.unsqueeze(1) * hidden_states + input_tensor
                modulated_part2 = gate2.unsqueeze(1) * hidden_states + input_tensor

                hidden_states = mask * modulated_part1 + (1 - mask) * modulated_part2
            else:
                hidden_states = gate.unsqueeze(1) * hidden_states + input_tensor
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class AGFMOutput(EsmOutput):
    def forward(self, hidden_states, input_tensor, gate=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if gate is not None:
            if isinstance(gate, list):
                gate1, gate2 = gate
                n = hidden_states.shape[1]
                n_half = n // 2

                mask = torch.zeros_like(hidden_states)
                mask[:, :n_half, :] = 1

                modulated_part1 = gate1.unsqueeze(1) * hidden_states + input_tensor
                modulated_part2 = gate2.unsqueeze(1) * hidden_states + input_tensor

                hidden_states = mask * modulated_part1 + (1 - mask) * modulated_part2
            else:
                hidden_states = gate.unsqueeze(1) * hidden_states + input_tensor
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class AGFMAttentionCodeFP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = ModifiedEsmSelfAttention(config)
        self.output = AGFMSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            shift_msa=None,
            scale_msa=None,
            gate_msa=None,
            type_ids=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)

        if shift_msa is not None:
            if isinstance(shift_msa, list):
                hidden_states_ln = modulate_diff(hidden_states_ln, shift_msa, scale_msa)
            else:
                hidden_states_ln = modulate(hidden_states_ln, shift_msa, scale_msa)

        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            type_ids=type_ids,
        )

        if gate_msa is not None:
            attention_output = self.output(self_outputs[0], hidden_states, gate_msa)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class AGFMLayerCodeFP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = AGFMAttentionCodeFP(config)
        self.intermediate = EsmIntermediate(config)
        self.output = AGFMOutput(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if not config.use_diff_modulation:
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
            
        else:
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 12 * config.hidden_size, bias=True),
        )

        self.adaLN_modulation.apply(_init_module_weights)

        self.use_diff_modulation = getattr(config, "use_diff_modulation", False)
        self.use_func_cross_attn = getattr(config, "use_func_cross_attn", False)
        self.use_motif_struct_emb = getattr(config, "use_motif_struct_emb", False)
        self.use_static_scale = getattr(config, "use_static_scale", False)
        self.use_attention_store = getattr(config, "use_attention_store", False)
        self.use_go_null_token = getattr(config, "use_go_null_token", False)
        self.use_motif_head = getattr(config, "use_motif_head", False)

        self.use_only_struct = getattr(config, "use_only_struct", False)

        if self.use_func_cross_attn:

            self.func_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.cross_attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                batch_first=True,
            )

            if not self.use_static_scale:
                self.cross_res_scale = nn.Parameter(torch.tensor(1.0))
            else:
                self.cross_res_scale = torch.tensor(1.0)

            self.cross_attn.apply(_init_module_weights)
            self.func_proj.apply(_init_module_weights)
            self.cross_attn_ln.apply(_init_module_weights)

        if self.use_motif_struct_emb:

            self.motif_proj = nn.Linear(1280, config.hidden_size, bias=True)
            self.motif_cross_attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.motif_cross_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                batch_first=True,
            )
            if not self.use_static_scale:
                self.motif_cross_res_scale = nn.Parameter(torch.tensor(0.1))
            else:
                self.motif_cross_res_scale = torch.tensor(0.2)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cond_input=None,
            type_ids=None,
            motif_struct_emb=None,
            go_type_mask=None,
    ):

        if cond_input is not None:
            if self.use_diff_modulation:

                if self.use_attention_store:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa_seq, scale_msa_seq, gate_msa_seq, shift_mlp_seq, scale_mlp_seq, gate_mlp_seq = self.adaLN_modulation(cond_input.sum(dim=1)).chunk(12, dim=1)
                else:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa_seq, scale_msa_seq, gate_msa_seq, shift_mlp_seq, scale_mlp_seq, gate_mlp_seq = self.adaLN_modulation(cond_input).chunk(12, dim=1)

                shift_msa = [shift_msa, shift_msa_seq]
                scale_msa = [scale_msa, scale_msa_seq]
                gate_msa = [gate_msa, gate_msa_seq]
                shift_mlp = [shift_mlp, shift_mlp_seq]
                scale_mlp = [scale_mlp, scale_mlp_seq]
                gate_mlp = [gate_mlp, gate_mlp_seq]
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond_input).chunk(6, dim=1)

        else:
            shift_msa = scale_msa = shift_mlp = scale_mlp = gate_msa = gate_mlp = None

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            shift_msa=shift_msa,
            scale_msa=scale_msa,
            gate_msa=gate_msa,
            type_ids=type_ids,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        cross_out = torch.zeros_like(attention_output)
        motif_cross_out = torch.zeros_like(attention_output)

        if self.use_func_cross_attn:
            if cond_input is not None:
                if cond_input.dim() == 2:
                    func_tok = cond_input.unsqueeze(1)
                else:
                    func_tok = cond_input

                func_tok = self.func_proj(func_tok)

                q = self.cross_attn_ln(attention_output)

                cross_out, cross_w = self.cross_attn(
                    query=q, key=func_tok, value=func_tok,
                    attn_mask=None, key_padding_mask=None,
                    need_weights=True,
                )

                if output_attentions:
                    outputs = (cross_w,) + outputs

        if self.use_motif_struct_emb:
            if motif_struct_emb is not None:
                if self.training and motif_struct_emb is not None:
                    dropout_prob = 0.1
                    if torch.rand(1).item() < dropout_prob:
                        motif_struct_emb = torch.zeros_like(motif_struct_emb)

                if motif_struct_emb.dim() == 2:
                    motif_tok = motif_struct_emb.unsqueeze(1)
                else:
                    motif_tok = motif_struct_emb

                motif_tok = self.motif_proj(motif_tok)

                q = self.motif_cross_attn_ln(attention_output)

                if motif_struct_emb.dim() == 2:
                    raise NotImplementedError("motif_struct_emb.dim() == 2")
                else:

                    motif_mask = (motif_struct_emb.sum(dim=-1) == 0)
                    all_motif_dropped = motif_mask.all(dim=1)

                    if motif_mask.all():
                        motif_cross_out = torch.zeros_like(q)
                        motif_cross_w = None
                    else:
                        motif_cross_out = torch.zeros_like(q)
                        motif_cross_w = torch.zeros_like(q)

                        valid_indices = ~all_motif_dropped
                        
                        if valid_indices.any():
                            valid_q = q[valid_indices]
                            valid_motif_tok = motif_tok[valid_indices]
                            valid_mask = motif_mask[valid_indices]

                            valid_out, valid_w = self.motif_cross_attn(
                                query=valid_q, 
                                key=valid_motif_tok, 
                                value=valid_motif_tok,
                                need_weights=output_attentions,
                                key_padding_mask=valid_mask
                            )

                            motif_cross_out = torch.zeros_like(q, dtype=valid_out.dtype)
                            motif_cross_out[valid_indices] = valid_out
                            if output_attentions:
                                motif_cross_w[valid_indices] = valid_w

                if output_attentions:
                    outputs = (motif_cross_w,) + outputs
            else:
                print("motif cond activate but no motif_struct_emb in input data")
                raise NotImplementedError
        
        n_half = attention_output.size(1) // 2
        struct_mask = torch.zeros_like(attention_output)
        struct_mask[:, :n_half, :] = 1

        if self.use_func_cross_attn:
            attention_output = attention_output + self.cross_res_scale * cross_out
        if self.use_motif_struct_emb:
            attention_output = attention_output + self.motif_cross_res_scale * motif_cross_out

        attention_output_ln = self.LayerNorm(attention_output)

        if cond_input is not None:
            if self.use_diff_modulation:
                attention_output_ln = modulate_diff(attention_output_ln, shift_mlp, scale_mlp)
            else:
                attention_output_ln = modulate(attention_output_ln, shift_mlp, scale_mlp)
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output, gate_mlp)
        else:
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs


class FuncTagEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        use_cfg_embedding = True
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class CodeFPEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

        self.use_go, self.use_ipr, self.use_ec = config.use_go, config.use_ipr, config.use_ec

        self.use_go_null_token = config.use_go_null_token

        if self.use_go:
            self.go_class_num = config.go_num
            self.go_cls_dropout_all = config.go_drop
            self.go_cls_dropout_each = 0.1
            self.go_embedder = FuncTagEmbedder(config.go_num, config.hidden_size)
            self.go_embedder.apply(_init_module_weights)
            if self.use_go_null_token:
                self.go_null_token = nn.Parameter(torch.zeros(config.hidden_size))
                self.go_null_token.data.normal_(mean=0.0, std=0.02)

        if self.use_ipr:
            self.ipr_class_num = config.ipr_num
            self.ipr_cls_dropout_all = config.ipr_drop
            self.ipr_cls_dropout_each = 0.1
            self.ipr_embedder = FuncTagEmbedder(config.ipr_num, config.hidden_size)
            self.ipr_embedder.apply(_init_module_weights)

        if self.use_ec:
            self.ec_class_num = config.ec_num
            self.ec_cls_dropout_all = config.ec_drop
            self.ec_cls_dropout_each = 0
            self.ec_embedder = FuncTagEmbedder(config.ec_num, config.hidden_size)
            self.ec_embedder.apply(_init_module_weights)

        self.layer = nn.ModuleList([AGFMLayerCodeFP(deepcopy(config)) for _ in range(config.num_hidden_layers)])

        if config.use_seq_motif and False:
            self.copy_blocks_num = config.num_hidden_layers//2
            self.anno_dropout = 0.5
            self.seq_controlnet = nn.ModuleList(
                [RCFEBlock(AGFMLayerCodeFP(deepcopy(config)), i, config.hidden_size) for i in range(self.copy_blocks_num)]
            )
        else:
            self.seq_controlnet = None

    def drop_anno_ids_add(self, class_tensor, embedder, class_num, training, drop_all_prob, drop_each_prob):
        """
        Drop annotation class IDs either at sample level or element level, then compute embeddings.
        """
        if training:
            drop_all = torch.rand(class_tensor.size(0), device=class_tensor.device) < drop_all_prob
            full_replacement = torch.full_like(class_tensor, class_num)
            class_tensor = torch.where(drop_all.unsqueeze(1), full_replacement, class_tensor)

            drop_each = torch.rand_like(class_tensor, dtype=torch.float) < drop_each_prob
            class_tensor = torch.where(drop_each, full_replacement, class_tensor)

        b_size = class_tensor.size(0)
        class_embeds = []
        if self.use_go_null_token:
            raise NotImplementedError("use_go_null_token not implemented")
            class_embeds.append(self.go_null_token.repeat(b_size, 1))

        for i, class_split in enumerate(class_tensor.split(1, dim=-1)):
            class_ids = class_split.squeeze(-1)
            class_embed = embedder(class_ids)
            mask = (class_ids == class_num).unsqueeze(-1)
            class_embed = torch.where(mask, torch.zeros_like(class_embed), class_embed)
            class_embeds.append(class_embed)

        return torch.sum(torch.stack(class_embeds, dim=0), dim=0)

    def drop_anno_ids_stack(self, class_tensor, embedder, class_num, training, drop_all_prob, drop_each_prob):
            """
            Drop annotation class IDs either at sample level or element level, then compute embeddings.
            
            【修改点】不再执行求和，而是返回一个序列 [B, Num_Tags, D]。
            """
            if training:
                drop_all = torch.rand(class_tensor.size(0), device=class_tensor.device) < drop_all_prob
                full_replacement = torch.full_like(class_tensor, class_num)
                class_tensor = torch.where(drop_all.unsqueeze(1), full_replacement, class_tensor)

                drop_each = torch.rand_like(class_tensor, dtype=torch.float) < drop_each_prob
                class_tensor = torch.where(drop_each, full_replacement, class_tensor)

            class_embeds = []
            for i, class_split in enumerate(class_tensor.split(1, dim=-1)):
                class_ids = class_split.squeeze(-1)

                if self.use_go_null_token:
                    mask_null_token = (class_ids == -2)
                    class_ids = torch.where(mask_null_token, class_num, class_ids)
                    mask_null_token = mask_null_token.unsqueeze(-1)
                
                class_embed = embedder(class_ids)
                mask = (class_ids == class_num).unsqueeze(-1)
                class_embed = torch.where(mask, torch.zeros_like(class_embed), class_embed)

                if self.use_go_null_token:
                    class_embed = torch.where(mask_null_token, self.go_null_token.repeat(class_embed.size(0), 1), class_embed)

                class_embeds.append(class_embed)

            return torch.stack(class_embeds, dim=1)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            type_ids=None,
            go_type_mask=None,
            **kwargs
    ):

        anno_tag = kwargs.get('anno_tag')
        anno_embed = None
        motif_struct_emb = None

        if anno_tag is not None:

            go_class = anno_tag.get('go')
            ipr_class = anno_tag.get('ipr')
            ec_class = anno_tag.get('ec')

            motif_struct_emb = anno_tag.get('motif_struct_emb')

            seq_num = hidden_states.size(0)

            def prepare_class(cls, class_num):
                if not self.training and cls.dim() == 1:
                    cls = cls.unsqueeze(0).repeat(seq_num, 1)
                return torch.where(cls == -1, torch.full_like(cls, class_num), cls)

            if not self.config.use_attention_store:

                if self.use_go and go_class is not None:
                    if hasattr(self.go_embedder, 'original_module'):
                        num_classes = self.go_embedder.original_module.num_classes
                    else:
                        num_classes = self.go_embedder.num_classes
                    go_class = prepare_class(go_class, num_classes)
                    anno_embed = self.drop_anno_ids_add(go_class, self.go_embedder, self.go_class_num,
                                                    self.training, self.go_cls_dropout_all, self.go_cls_dropout_each)

                if self.use_ipr and ipr_class is not None:
                    if hasattr(self.ipr_embedder, 'original_module'):
                        num_classes = self.ipr_embedder.original_module.num_classes
                    else:
                        num_classes = self.ipr_embedder.num_classes
                    ipr_class = prepare_class(ipr_class, num_classes)
                    ipr_embed = self.drop_anno_ids_add(ipr_class, self.ipr_embedder, self.ipr_class_num,
                                                self.training, self.ipr_cls_dropout_all, self.ipr_cls_dropout_each)
                    anno_embed = anno_embed + ipr_embed if anno_embed is not None else ipr_embed

                if self.use_ec and ec_class is not None:
                    if hasattr(self.ec_embedder, 'original_module'):
                        num_classes = self.ec_embedder.original_module.num_classes
                    else:
                        num_classes = self.ec_embedder.num_classes
                    ec_class = prepare_class(ec_class, num_classes)
                    ec_embed = self.drop_anno_ids_add(ec_class, self.ec_embedder, self.ec_class_num,
                                                self.training, self.ec_cls_dropout_all, self.ec_cls_dropout_each)
                    anno_embed = anno_embed + ec_embed if anno_embed is not None else ec_embed

            else:
                print("use attention store")
                anno_embeds = []

                if self.use_go and go_class is not None:
                    if hasattr(self.go_embedder, 'original_module'):
                        num_classes = self.go_embedder.original_module.num_classes
                    else:
                        num_classes = self.go_embedder.num_classes
                    go_class = prepare_class(go_class, num_classes)
                    go_embed = self.drop_anno_ids_stack(go_class, self.go_embedder, self.go_class_num,
                                                    self.training, self.go_cls_dropout_all, self.go_cls_dropout_each)
                    anno_embeds.append(go_embed)

                if self.use_ipr and ipr_class is not None:
                    if hasattr(self.ipr_embedder, 'original_module'):
                        num_classes = self.ipr_embedder.original_module.num_classes
                    else:
                        num_classes = self.ipr_embedder.num_classes
                    ipr_class = prepare_class(ipr_class, num_classes)
                    ipr_embed = self.drop_anno_ids_stack(ipr_class, self.ipr_embedder, self.ipr_class_num,
                                                    self.training, self.ipr_cls_dropout_all, self.ipr_cls_dropout_each)
                    anno_embeds.append(ipr_embed)

                if self.use_ec and ec_class is not None:
                    if hasattr(self.ec_embedder, 'original_module'):
                        num_classes = self.ec_embedder.original_module.num_classes
                    else:
                        num_classes = self.ec_embedder.num_classes
                    ec_class = prepare_class(ec_class, num_classes)
                    ec_embed = self.drop_anno_ids_stack(ec_class, self.ec_embedder, self.ec_class_num,
                                                    self.training, self.ec_cls_dropout_all, self.ec_cls_dropout_each)
                    anno_embeds.append(ec_embed)

                if anno_embeds:
                    anno_embed = torch.cat(anno_embeds, dim=1)            

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None

        output_attentions = output_attentions or (self.config.use_attention_store)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        if self.seq_controlnet and anno_tag['seq_cond'] is not None and anno_tag['seq_cond'].numel() > 0 and False:

            motif = anno_tag['seq_cond']
          
            random_go_embed = anno_embed if (not self.training or random.random() > self.anno_dropout) else None

            for index in range(1, self.copy_blocks_num + 1):
                motif, motif_skip = self.seq_controlnet[index - 1](hidden_states, attention_mask, motif, random_go_embed)
                hidden_states = self.layer[index](hidden_states+motif_skip, attention_mask, cond_input=random_go_embed)[0]

            for index in range(self.copy_blocks_num + 1, len(self.layer)):
                hidden_states = self.layer[index](hidden_states, attention_mask, cond_input=random_go_embed)[0]

        else:

            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    print(f"Bad gradient_checkpointing: {self.gradient_checkpointing}")
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        anno_embed,
                        type_ids,
                        motif_struct_emb,
                        go_type_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        anno_embed,
                        type_ids,
                        motif_struct_emb,
                        go_type_mask,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ModifiedEsmModelCodeFP(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = CodeFPEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        self.hidden_size = config.hidden_size

        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            seq_cond_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            type_ids: Optional[torch.Tensor] = None,
            go_type_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            try:
                input_shape = input_ids['x_t'].size()
            except (KeyError, TypeError, AttributeError, IndexError):
                input_shape = input_ids.size() if torch.is_tensor(input_ids) else None
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        if not torch.is_tensor(input_ids):
            device = input_ids['x_t'].device if input_ids is not None else inputs_embeds.device
        else:
            device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        elif attention_mask.dim() == 2:
            extended_attention_mask: torch.Tensor = (
                self.get_extended_attention_mask(attention_mask, input_shape)
            )
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})! "
                "Should be [batch_size, seq_length] or [batch_size, seq_length, seq_length]."
            )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids if torch.is_tensor(input_ids) else input_ids['x_t'],
            position_ids=position_ids,
            attention_mask=input_ids['x_t'].ne(
                self.config.pad_token_id
            ),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            anno_tag=input_ids,
            type_ids=type_ids,
            go_type_mask=go_type_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class EsmForCodeFP(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        # print(f"Loading model from {config._name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = ModifiedEsmModelCodeFP(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()
        
        self.pad_id = tokenizer.pad_token_id
        self.config.pad_token_id = self.pad_id
        
        self.contact_head = None
        self.tokenizer = tokenizer
    
    def forward(self,
                input_ids,
                attention_mask=None,
                type_ids=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                go_type_mask=None,
            ):

        assert isinstance(input_ids, dict)

        seq_cond_attention_mask = None

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            seq_cond_attention_mask=seq_cond_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            type_ids=type_ids,
            go_type_mask=go_type_mask,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        return result


def get_net(cfg):

    if cfg.net.arch_type == 'func_esm_dplm2':
        # print(f"net name: {cfg.net.name}")
        config = AutoConfig.from_pretrained(f'{cfg.net.name}')
        config.hidden_dropout_prob = cfg.net.dropout
        config.tie_word_embeddings = False
        config.vocab_size = 8229

        config.use_diff_modulation = getattr(cfg, "use_diff_modulation", False)
        config.use_func_cross_attn = getattr(cfg, "use_func_cross_attn", False)
        config.use_motif_struct_emb = getattr(cfg, "use_motif_struct_emb", False)
        config.use_static_scale = getattr(cfg, "use_static_scale", False)
        config.use_attention_store = getattr(cfg, "use_attention_store", False)
        config.use_go_null_token = getattr(cfg, "use_go_null_token", False)
        config.use_motif_head = getattr(cfg, "use_motif_head", False)

        config.use_only_struct = getattr(cfg, "use_only_struct", False)

        cond = getattr(cfg, "cond", None)
        if cond is not None:
            config.update(cond.todict() if hasattr(cond, "todict") else cond)
        net = EsmForCodeFP(config, dropout=cfg.net.dropout)
    else:
        raise NotImplementedError
        
    if cfg.lora.lora:
        raise NotImplementedError

    return net


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def topk_masking_prior(
    scores, cutoff_len, stochastic=False, temp=1.0, prior_mask=None
):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(
        dim=-1, index=cutoff_len
    )
    masking = _scores < cutoff
    return masking


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    return tokens, scores


def top_k_top_p_filtering(logits, top_k=0, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_logits[sorted_indices_to_remove] = filter_value
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape) 
    return logits

class MotifLabelHead(nn.Module):
    """Head for predicting the aggregated Motif Label (Classification)."""
    def __init__(self, hidden_size, num_motif_labels):
        super().__init__()
        # 使用一个线性层将 [CLS] 状态 (config.hidden_size) 映射到标签类别数
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_motif_labels)
        )
        self.num_motif_labels = num_motif_labels

    def forward(self, cls_hidden_state):
        # cls_hidden_state 形状: [B, D]
        logits = self.classifier(cls_hidden_state) # [B, num_motif_labels]
        return logits

class CodeFPFunctionalProteinDesignModel(nn.Module):
    _default_cfg = CodeFPConfig()
    
    def __init__(self, cfg, net=None):
        # print("init cdplm2")
        super().__init__()
        # print("init cdplm2")
        # print(str(cfg))
        self._update_cfg(cfg)
        # print("init cdplm2")
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file,
        )
        self._prepare_special_token()
        self.cfg.tokenizer.vocab_size = len(self.tokenizer)
        # print("init cdplm2")

        # Note：嵌入很深，触及到esm.encode层的注意力机制，小心改
        self.net = get_net(cfg) if net is None else net
        # self.tokenizer = self.net.tokenizer
        # print("init cdplm2")


        self.use_motif_head = getattr(self.cfg, 'use_motif_head', False)
        if self.use_motif_head:
            self.motif_head = MotifLabelHead(
                1280, self.cfg.cond.go_num
            )
            self._init_weights(self.motif_head)
            


        self.use_diff_ce = getattr(self.cfg, 'use_diff_ce', False)
        self.use_motif_struct_emb = getattr(self.cfg, 'use_motif_struct_emb', False)
        self.use_static_scale = getattr(self.cfg, 'use_static_scale', False)

        self.use_attention_store = getattr(self.cfg, 'use_attention_store', False)
        # self.use_go_null_token = getattr(self.cfg, 'use_go_null_token', False) # ???
        self.use_motif_head = getattr(self.cfg, 'use_motif_head', False)

        self.use_only_struct = getattr(self.cfg, 'use_only_struct', False)

        
        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()
    
        self._struct_tokenizer = None

        if self.use_attention_store:
            raise NotImplementedError

        # print("init cdplm2 done")

    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def _prepare_special_token(self):
        self.aa_bos_id = self.tokenizer._token_to_id["<cls_aa>"]
        self.aa_eos_id = self.tokenizer._token_to_id["<eos_aa>"]
        self.aa_mask_id = self.tokenizer._token_to_id["<mask_aa>"]
        self.struct_bos_id = self.tokenizer._token_to_id["<cls_struct>"]
        self.struct_eos_id = self.tokenizer._token_to_id["<eos_struct>"]
        self.struct_mask_id = self.tokenizer._token_to_id["<mask_struct>"]
        self.pad_id = self.tokenizer._token_to_id["<pad>"]
        self.aa_unk_id = self.tokenizer._token_to_id["<unk_aa>"]
        self.struct_unk_id = self.tokenizer._token_to_id["<unk_struct>"]

        self.aa_X_id = self.tokenizer._token_to_id["X"]
        self.aa_B_id = self.tokenizer._token_to_id["B"]
        self.aa_U_id = self.tokenizer._token_to_id["U"]
        self.aa_Z_id = self.tokenizer._token_to_id["Z"]
        self.aa_O_id = self.tokenizer._token_to_id["O"]

        self.aa_type = 1
        self.struct_type = 0
        self.pad_type = 2

    @property
    def special_token_list(self):
        return [
            self.aa_bos_id,
            self.aa_eos_id,
            self.aa_mask_id,
            self.struct_bos_id,
            self.struct_eos_id,
            self.struct_mask_id,
            self.pad_id,
            self.aa_unk_id,
            self.struct_unk_id,
            self.aa_X_id,
            self.aa_B_id,
            self.aa_U_id,
            self.aa_Z_id,
            self.aa_O_id,
        ]

    @classmethod
    def from_pretrained(cls, net_name, cfg_override={}, net_override={}, from_huggingface=False):
        if not from_huggingface:
            # Load model checkpoint from local if you pretrain a DPLM with this repo
            # The net_name should be like:
            # ${name}/checkpoints/last.ckpt
            # and there should be .hydra/config.yaml in the ${name} directory that is automatically generated during training.
            def load_yaml_config(fpath: str) -> OmegaConf:
                return OmegaConf.load(fpath)
            from pathlib import Path
            from collections import OrderedDict
            
            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, '.hydra', 'config.yaml')
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop('_target_')
            model = cls(cfg)
            
            pretrained_state_dict = torch.load(net_name, map_location=torch.device("cpu"))['state_dict']
            new_pretrained_state_dict = OrderedDict()
            
            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v

# 打印模型实际的参数名
            # model_params = set(model.state_dict().keys())
            # # 打印权重文件里的参数名
            # ckpt_params = set(new_pretrained_state_dict.keys())

            model.load_state_dict(new_pretrained_state_dict, strict=True)

            # def count_all_parameters(model):
            #     return sum(p.numel() for p in model.parameters())

            # total_params = count_all_parameters(model)

            return model
        else:
            raise NotImplementedError

    @property
    def device(self):
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device("cpu")
        return device

    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        raise NotImplementedError


    def q_sample(self, x_0, t, type_ids, maskable_mask):
        aa_position = type_ids == self.aa_type
        struct_position = type_ids == self.struct_type

        # sample x_t
        u = torch.rand_like(x_0, dtype=torch.float)
        t_mask = (
            u < (t / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t = x_0.masked_fill(t_mask & aa_position, self.aa_mask_id)
        x_t = x_t.masked_fill(t_mask & struct_position, self.struct_mask_id)

        return x_t, t_mask
        
    def get_modality_type(self, input_ids):
        input_mask = input_ids.ne(self.pad_id)
        # HACK: all amino acid token id < 33, while all struct token id >= 33
        # 0 stands for struct, 1 stands for aa
        modality_type = ((input_ids < 33) & input_mask).int()
        # 2 stands for padding
        modality_type[~input_mask] = self.pad_type
        return modality_type
    
    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):

        input_mask = input_ids['x_t'].ne(self.pad_id)

        type_ids = self.get_modality_type(input_ids['x_t'])

        L = input_ids['x_t'].shape[1]
        num_heads = self.net.config.num_attention_heads
        # [B, num_heads, L+2, L+2]
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, input_ids['x_t'].shape
            ).repeat(1, num_heads, L, 1)
        )  # -inf for padding positions, 0 otherwise

        if "single_modality" in kwargs:
            single_modality_index = kwargs["single_modality"]
            struct_attention_bias, aa_attention_bias = attention_bias.chunk(
                2, dim=-2
            )
            struct_attention_bias[
                single_modality_index, :, :, L // 2 :
            ] = -math.inf
            aa_attention_bias[
                single_modality_index, :, :, : L // 2
            ] = -math.inf
            attention_bias = torch.concat(
                [struct_attention_bias, aa_attention_bias], dim=-2
            )

        # [B, L, d_model]
        # input_embeds = self.net.esm.embeddings(
        #     input_ids, attention_mask=input_mask
        # )
        input_embeds = None

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            type_ids=type_ids,
            go_type_mask=input_ids['go_type_mask'],
        )


        return outputs       


    def get_motif_original(self, target, motif_start_end, motif_len_min, motif_len_max, min_mask_ratio=0.05, max_mask_ratio=0.1):
        batch_size, sequence_length = target.shape
        masked_targets = []

        for i in range(batch_size):
            current_target = target[i].clone()

            non_special_sym_mask = (
                    (current_target != self.pad_id) &
                    (current_target != self.bos_id) &
                    (current_target != self.eos_id)
            )
            effective_indices = torch.where(non_special_sym_mask)[0]

            if len(effective_indices) == 0:
                masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                continue

            total_length = len(effective_indices)
            retain_min_len = max(motif_len_min, int(min_mask_ratio * total_length))
            retain_max_len = max(motif_len_max, int(max_mask_ratio * total_length))

            start, end = motif_start_end[i]

            if start == 0 and end == 0:
                retain_length = torch.randint(retain_min_len, retain_max_len + 1, (1,)).item()
                retain_start_idx = torch.randint(0, total_length - retain_length + 1, (1,)).item()
                retain_start = effective_indices[retain_start_idx].item()
                retain_end = effective_indices[retain_start_idx + retain_length - 1].item()
            else:
                motif_length = end - start
                if motif_length < retain_min_len:
                    retain_length = retain_min_len
                elif motif_length > retain_max_len:
                    retain_length = retain_max_len
                else:
                    retain_length = motif_length

                if end - start - retain_length > 0:
                    retain_start = torch.randint(start, end - retain_length + 1, (1,)).item()
                else:
                    retain_start = start

                retain_end = retain_start + retain_length - 1

            sequence_indices = torch.arange(sequence_length, device=target.device)
            mask = non_special_sym_mask & ((sequence_indices < retain_start) | (sequence_indices > retain_end))
            masked_target = current_target.clone()
            masked_target[mask] = self.mask_id

            masked_targets.append(masked_target)

        return torch.stack(masked_targets)

    def get_motif_middle(self, target, motif_start_end, motif_len_min=10, motif_len_max=30):

        batch_size, sequence_length = target.shape
        masked_targets = []

        for i in range(batch_size):
            current_target = target[i].clone()
            if sum(motif_start_end[i]) == 0:
                non_special_sym_mask = (
                        (current_target != self.pad_id) &
                        (current_target != self.bos_id) &
                        (current_target != self.eos_id)
                )
                effective_indices = torch.where(non_special_sym_mask)[0]
                if len(effective_indices) == 0:
                    masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                    continue

                start = effective_indices[0].item()
                end = effective_indices[-1].item()
            else:
                start, end = motif_start_end[i]

            motif_length = end - start

            if motif_length < motif_len_min:
                crop_len = motif_length
            else:
                crop_len = min(torch.randint(motif_len_min, min(motif_len_max, motif_length) + 1, (1,)).item(), motif_length)

            non_special_sym_mask = (
                    (current_target != self.pad_id) &
                    (current_target != self.bos_id) &
                    (current_target != self.eos_id)
            )

            effective_indices = torch.where(non_special_sym_mask)[0]
            if len(effective_indices) == 0:
                masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                continue

            middle_position = (effective_indices[0] + effective_indices[-1]) // 2
            crop_start = max(middle_position - crop_len // 2, effective_indices[0])
            crop_end = min(crop_start + crop_len, effective_indices[-1] + 1)
            crop_start = crop_end - crop_len

            masked_target = current_target.clone()
            masked_target[non_special_sym_mask] = self.mask_id
            masked_target[crop_start:crop_end] = current_target[crop_start:crop_end]

            masked_targets.append(masked_target)

        masked_target = torch.stack(masked_targets)

        return masked_target

    def construct_x_t(self, struct_target, aatype_target, struct_ignore=None, seq_ignore=False):
        bsz = struct_target.size(0)
        # seperately add noise to struct and aa
        struct_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=struct_target.device,
        )
        aatype_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=aatype_target.device,
        )

        assert (
            self.cfg.single_modality_ratio
            + self.cfg.folding_loss_ratio
            + self.cfg.inverse_folding_loss_ratio
            + self.cfg.joint_loss_ratio
            + self.cfg.independent_loss_ratio
            == 1.0
        )

        split_sizes = [
            int(bsz * self.cfg.single_modality_ratio),
            int(bsz * self.cfg.folding_loss_ratio),
            int(bsz * self.cfg.inverse_folding_loss_ratio),
            int(bsz * self.cfg.independent_loss_ratio),
            int(bsz * self.cfg.joint_loss_ratio),
        ]
        split_sizes[-1] = bsz - sum(split_sizes[:-1])

        rand_index = torch.randperm(bsz).type_as(struct_target)
        int_index_list = torch.split(rand_index, split_sizes)

        bool_index_list = []
        for int_index in int_index_list:
            bool_index = torch.zeros(bsz, dtype=torch.bool).to(
                struct_target.device
            )
            bool_index[int_index] = True

            if struct_ignore is not None:
                bool_index = bool_index & ~struct_ignore

            bool_index_list.append(bool_index)

        (
            single_modality_index,
            folding_index,
            inverse_folding_index,
            independent_index,
            joint_index,
        ) = bool_index_list

        struct_t = struct_t.masked_fill(inverse_folding_index, 0)

        if struct_ignore is not None:
            struct_t = torch.where(
                struct_ignore,
                torch.tensor(self.cfg.num_diffusion_timesteps, device=struct_t.device).expand_as(struct_t),
                struct_t
            )


        struct_type_id = self.get_modality_type(struct_target)
        struct_x_t, struct_loss_mask = self.q_sample(
            struct_target,
            struct_t,
            struct_type_id,
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])

        if seq_ignore:
            aatype_t[:] = self.cfg.num_diffusion_timesteps

        aa_type_id = self.get_modality_type(aatype_target)
        aatype_x_t, aa_loss_mask = self.q_sample(
            aatype_target,
            aatype_t,
            aa_type_id,
            maskable_mask=self.get_non_special_symbol_mask(aatype_target),
        )

        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def get_motif_hidden_states_and_labels(self, hidden_states, motif_position_and_label_list):
        """
        根据motif位置信息提取对应的hidden states和labels
        
        Args:
            hidden_states: list of [seq_len, hidden_dim], batch中的每个序列的hidden states
            motif_position_and_label_list: list of dict, 每个dict包含motif的位置和标签信息
        
        Returns:
            motif_hidden_states: [total_motifs, hidden_dim]
            motif_labels: [total_motifs]
        """
        all_motif_states = []
        all_motif_labels = []
        
        for i, (hidden_state, motif_dict) in enumerate(zip(hidden_states, motif_position_and_label_list)):
            # hidden_state: [seq_len, hidden_dim]
            for (start, end), label in motif_dict.items():

                if end - start <= 5:
                    print(f"bad start:{start} end:{end} label:{label}")
                    continue
                # 提取motif区域的hidden states
                motif_region = hidden_state[start:end]  # [motif_len, hidden_dim]

                if torch.isnan(motif_region).any():
                    print(f"motif_region contains nan: {motif_region}")
                    print(f"start: {start}, end: {end}, label: {label}")
                    print(f"hidden_state: {hidden_state}")
                    print(f"shape hidden_state: {hidden_state.shape}")
                    raise ValueError("motif_region contains nan")
                
                # 对motif区域进行池化（平均池化）
                motif_embedding = torch.mean(motif_region, dim=0)  # [hidden_dim]

                if torch.isnan(motif_embedding).any():
                    print(f"motif_embedding contains nan: {motif_embedding}")
                    print(f"start: {start}, end: {end}, label: {label}")
                    print(f"motif_region: {motif_region}")
                    raise ValueError("motif_embedding contains nan")
                
                all_motif_states.append(motif_embedding)
                all_motif_labels.append(label)
        
        if len(all_motif_states) == 0:
            # 如果没有motif，返回空tensor
            return torch.tensor([], device=hidden_states[0].device), torch.tensor([], device=hidden_states[0].device, dtype=torch.long)
        
        motif_hidden_states = torch.stack(all_motif_states)  # [total_motifs, hidden_dim]
        motif_labels = torch.tensor(all_motif_labels, device=motif_hidden_states.device, dtype=torch.long)

        if torch.isnan(motif_hidden_states).any():
            print(f"motif_hidden_states contains nan: {motif_hidden_states}")
            raise ValueError("motif_hidden_states contains nan")
        
        return motif_hidden_states, motif_labels


    
    def compute_loss(self, batch, weighting='constant'):
        target = batch['targets']

        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(struct_target, aatype_target, struct_ignore=batch.get('struct_ignore', None), seq_ignore=self.use_only_struct)

        # print(aatype_noised)
        # exit()

        x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)

        masked_target = None
        if self.cfg.cond.use_seq_motif:
            if random.random() < 0.5:
                masked_target = self.get_motif_original(target, batch['motif_start_end'], motif_len_min=self.cfg.cond.motif_min_len, motif_len_max=self.cfg.cond.motif_max_len)
            else:
                masked_target = self.get_motif_middle(target, batch['motif_start_end'], motif_len_min=self.cfg.cond.motif_min_len, motif_len_max=self.cfg.cond.motif_max_len)

        motif_struct_emb = None
        if self.use_motif_struct_emb:
            # print(f"use motif_struct_emb")
            motif_struct_emb = batch['motif_struct_emb']
            # print(motif_struct_emb)

        inputs = dict(x_t=x_t, seq_cond=masked_target, go=batch['go_type'], ipr=batch['ipr_type'], ec=batch['ec_type'], motif_struct_emb=motif_struct_emb, go_type_mask=batch.get('go_type_mask', None))

        # if batch.get("struct_ignore") is not None:
        #    temp = batch['struct_ignore']
        #    if sum(temp) > 0:
        #        print("ignore struct", struct_noised["t"])
               

        model_outputs = self.forward(
                input_ids=inputs,
                single_modality=single_modality_index,
            )



        struct_logits, aatype_logits = model_outputs["logits"].chunk(2, dim=1)
        struct_hidden_state, aatype_hidden_state = model_outputs["last_hidden_state"].chunk(2, dim=1)


        motif_labels_logits = None
        motif_labels_target = None
        if self.use_motif_head:
            features, labels = self.get_motif_hidden_states_and_labels(struct_hidden_state, batch.get("motif_position_and_label", None))

            if len(features) == 0:
                print(f"features: {features}")
                print(f"labels: {labels}")

                motif_labels_logits = torch.tensor([], device=struct_logits.device)
                motif_labels_target = torch.tensor([], device=struct_logits.device, dtype=torch.long)
            else:
                motif_labels_logits = self.motif_head(features)
                motif_labels_target = labels

        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_weight = {
            "linear": (
                num_timesteps - (struct_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        struct_weight = struct_weight.expand(struct_target.size())

        struct_weight_point = {
            "linear": (
                num_timesteps - (struct_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting].float() / num_timesteps

        aa_weight_point = {
            "linear": (
                num_timesteps - (aatype_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(aatype_noised["t"]),
        }[weighting].float() / num_timesteps

        # 将需要忽略的位置的权重设置为0
        if 'struct_ignore' in batch:
            # pass
            struct_ignore = batch['struct_ignore'].unsqueeze(1).expand(struct_target.size())
            struct_weight = struct_weight * (~struct_ignore).float()

        aatype_weight = {
            "linear": (
                num_timesteps - (aatype_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(aatype_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        aatype_weight = aatype_weight.expand(aatype_target.size())

        if self.use_only_struct:
            aatype_weight[:] = 0.0


        attn_map_loss_dict = None
        avg_attn_map = None
        if self.use_attention_store:
            # print(f"attention_store: {self.attention_store.attention_maps}")
            attenion_maps = self.attention_store.get_attention_maps()
            print(f"attn maps: {attenion_maps}")

            avg_attn_map = []
            for name, attn_map in attenion_maps.items():
                avg_attn_map.append(attn_map)
            # print(f"avg_attn_map: {avg_attn_map}")
            avg_attn_map = torch.stack(avg_attn_map).mean(0)
            # print(f"avg_attn_map: {avg_attn_map.shape}")
            print(f"avg_attn_map: {avg_attn_map}")

            avg_attn_map = avg_attn_map.permute(0, 2, 1)
            # print(f"avg_attn_map: {avg_attn_map.shape}")

            self.attention_store.reset()

            attn_map_loss_dict = {
                "attn_map": avg_attn_map,
                "struct_weight_point": struct_weight_point,
                "aatype_weight_point": aa_weight_point,
                "go_type_segments": batch.get("go_type_segments", None),
                "go_type_segments_mask": batch.get("go_type_segments_mask", None),
            }



        # just scale original factor
        if self.use_diff_ce:
            # print("in diff ce ")
            struct_t = struct_noised["t"].float()
            aatype_t = aatype_noised["t"].float()   
            motif_mask = batch['motif_mask']

            scale_alpha = 0.5

            max_alpha = 0.25
            times_gama = 3

            # 结构权重计算（基于struct_t）
            struct_time_factor = (struct_t - 1) / (num_timesteps)  # 归一化到[0,1]
            struct_motif_weight_coeff = scale_alpha * (1 + max_alpha - (max_alpha * torch.exp(-times_gama * struct_time_factor)))
            struct_motif_weight_coeff = struct_motif_weight_coeff[:, None].expand(-1, motif_mask.shape[1])

            # print(f"motif_mask: {motif_mask.shape}, struct_motif_weight_coeff: {struct_motif_weight_coeff.shape}, struct_weight: {struct_weight.shape}")
            # assert motif_mask.shape == struct_motif_weight_coeff.shape == struct_weight.shape

            # print(f"sum motif_mask: {motif_mask.sum()}, sum struct_motif_weight_coeff: {struct_motif_weight_coeff.sum()}, sum struct_weight: {struct_weight.sum()}")
            # 应用结构motif权重
            struct_weight_coeff = torch.where(~motif_mask, struct_motif_weight_coeff, scale_alpha)
            struct_weight = struct_weight * struct_weight_coeff

            # 氨基酸类型权重计算（基于aatype_t）
            aatype_time_factor = (aatype_t - 1) / (num_timesteps)  # 归一化到[0,1]
            aatype_motif_weight_coeff = scale_alpha * (1 + max_alpha - (max_alpha * torch.exp(-times_gama * aatype_time_factor)))
            aatype_motif_weight_coeff = aatype_motif_weight_coeff[:, None].expand(-1, motif_mask.shape[1])
            # 应用氨基酸类型motif权重
            aatype_weight_coeff = torch.where(~motif_mask, aatype_motif_weight_coeff, scale_alpha)
            aatype_weight = aatype_weight * aatype_weight_coeff
        # if self.use_diff_ce:
        #     # TODO 莫名其妙不需要改前向过程？原来的公式就不对，并且有部分t=0，比较奇怪
        #     # print("in diff ce ")
        #     # print(f"motif_mask: {batch['motif_mask'].shape}, struct_weight: {struct_weight.shape}, aatype_weight: {aatype_weight.shape}")
        #     # print(f"original struct_weight: {struct_weight}")
        #     struct_t = struct_noised["t"]
        #     aatype_t = aatype_noised["t"]
        #     motif_mask = batch['motif_mask']

        #     gamma = 2.0

        #     aatype_time_factor = (aatype_t - 1) / num_timesteps
        #     aatype_motif_weight = 1.0 - torch.pow(aatype_time_factor, gamma)
        #     aatype_motif_weight = aatype_motif_weight[:, None].float().expand(aatype_target.size())

        #     aatype_weight = torch.where(motif_mask, aatype_motif_weight, aatype_weight)

        #     struct_time_factor = (struct_t - 1) / num_timesteps
        #     struct_motif_weight = 1.0 - torch.pow(struct_time_factor, gamma)
        #     struct_motif_weight = struct_motif_weight[:, None].float().expand(struct_target.size())
        #     struct_weight = torch.where(motif_mask, struct_motif_weight, struct_weight)

        #     # print(f"final struct_weight: {struct_weight}")
        #     # exit()

        # if self.use_attention_store:
        #     attention_maps = self.attention_store.get_attention_maps()
        #     print(f"attention_maps: {attention_maps}")
        #     print(f"attention_maps.keys(): {attention_maps.keys()}")
        #     print(f"attention_maps.values(): {attention_maps.values()}")
        #     exit()
        return (
            {
                "aatype": aatype_logits,
                "struct": struct_logits,
            },  # model pred logits
            {
                "aatype": aatype_target,
                "struct": struct_target,
            },  # training targets
            {  # training loss mask
                "aatype": aatype_noised["mask"],
                "struct": struct_noised["mask"],
            },
            {
                "aatype": aatype_weight,
                "struct": struct_weight,
            },  # training loss weight
            {
                "aatype": aatype_hidden_state,
                "struct": struct_hidden_state,
                "motif": batch.get("motif_position_and_label", None),
                "struct_weight_point": struct_weight_point,
                
                "motif_labels_logits": motif_labels_logits,
                "motif_labels": motif_labels_target,
            },  # training hidden state
            attn_map_loss_dict,
        )

    def forward_encoder(self, batch, **kwargs):
        return {}

    def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
        non_special_symbol_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.aa_bos_id)
            & output_tokens.ne(self.aa_eos_id)
            & output_tokens.ne(self.struct_bos_id)
            & output_tokens.ne(self.struct_eos_id)
        )
        if partial_masks is not None:
            non_special_symbol_mask &= ~partial_masks
        return non_special_symbol_mask

    def initialize_output_tokens(
        self, input_tokens, partial_masks=None, **kwargs
    ):
        type_ids = self.get_modality_type(input_tokens)
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        # fill the aatype part and struct part with specialized mask token
        aa_position = type_ids.eq(self.aa_type) & output_mask
        struct_position = type_ids.eq(self.struct_type) & output_mask
        output_tokens = input_tokens.masked_fill(aa_position, self.aa_mask_id)
        output_tokens = output_tokens.masked_fill(
            struct_position, self.struct_mask_id
        )
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

        return output_tokens, output_scores

    def resample_conditional(self, _tokens, _scores, ratio, scale, go=None, ipr=None, seq_cond=None, ec=None, motif_struct_emb=None, **kwargs):
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []
        resample_input_seq_cond = []
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                
                token = int(token)
                if token == self.pad_id or token >= 33:
                    # just check aa
                    continue
                
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio * 0.5:#max(0.3/(step+1) ** 0.2, 0.1):
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio * 0.5:#max(0.3/(step+1) ** 0.2, 0.1):
                        mask |= seq.eq(k)
                # resample_input_mask.append(mask)
                # resample_input.append(seq.masked_fill(mask, self.aa_mask_id))

                seq = seq.masked_fill(mask, self.aa_mask_id)
                
                struct_mask = torch.zeros_like(seq).bool()
                for id, value in enumerate(mask):
                    if value:
                        struct_mask[id - (len(seq) // 2)] = value

                seq = seq.masked_fill(struct_mask, self.struct_mask_id)

                all_mask = struct_mask | mask
                resample_input_mask.append(all_mask)
                resample_input.append(seq)
                
                if seq_cond is not None:
                    raise NotImplementedError
                    # resample_input_seq_cond.append(seq_cond[i].masked_fill(mask, self.mask_id))
                #resample_input.append(seq.masked_scatter(mask, xt[i][mask]))
            
        if len(to_be_resample_idx) > 0:
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(_scores)
            resample_input_mask = torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            if seq_cond is not None:
                raise NotImplementedError
                resample_input_seq_cond = torch.stack(resample_input_seq_cond, dim=0).type_as(_tokens)
            if motif_struct_emb is not None:
                inputs = dict(x_t=resample_input, go=go, ipr=ipr, seq_cond=resample_input_seq_cond if seq_cond is not None else None, ec=ec, motif_struct_emb=motif_struct_emb[to_be_resample_idx])
            else:
                inputs = dict(x_t=resample_input, go=go, ipr=ipr, seq_cond=resample_input_seq_cond if seq_cond is not None else None, ec=ec, motif_struct_emb=None)
            type_ids = self.get_modality_type(_tokens)
            
            resample_logits = self.net(
                input_ids=inputs, type_ids=type_ids
            )['logits']
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)

            
            output_masks = self.get_non_special_symbol_mask(_tokens, partial_masks=None)

            aa_position = type_ids.eq(self.aa_type) & output_masks
            struct_position = type_ids.eq(self.struct_type) & output_masks
            indices_aa = torch.where(aa_position)
            indices_struct = torch.where(struct_position)

            # HACK: all amino acid token id < 33, while all struct token id >= 33
            resample_logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
            resample_logits[indices_struct[0], indices_struct[1], :33] = -math.inf

            resample_logits[..., self.special_token_list] = -math.inf

            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            #noise_scale = 1.5 - 0.2 * ((step + 1) / max_step)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(resample_logits, temperature=0.0, noise_scale=noise_scale)
            resample_input.masked_scatter_(resample_input_mask, resample_tokens[resample_input_mask])
            resample_input_scores.masked_scatter_(resample_input_mask, resample_scores[resample_input_mask])
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = resample_input, resample_input_scores
            
    def forward_decoder(self, prev_decoder_out, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='gumbel_argmax', go_label=None, ipr_label=None, seq_cond=None, ec_label=None, motif_struct_emb=None):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_symbol_mask(output_tokens, partial_masks=partial_masks)

        inputs = dict(x_t=output_tokens, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label, motif_struct_emb=motif_struct_emb)

        input_mask = output_tokens.ne(self.pad_id)
        L = output_tokens.shape[1]
        num_heads = self.net.config.num_attention_heads
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, output_tokens.shape
            ).repeat(1, num_heads, L, 1)
        )
        # print(attention_bias.shape)

        net_out = self.net(input_ids=inputs,attention_mask=attention_bias,type_ids=self.get_modality_type(output_tokens))

        # TODO: BUG?? 没取logsoftmax，检查要不要logsoftmax，后续处理也有存在logsoftmax的地方
        # 但是不影响
        logits = net_out['logits']
        # logits = net_out["logits"].log_softmax(dim=-1)
        attentions = net_out['attentions'] if need_attn_weights else None
        
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        type_ids = self.get_modality_type(output_tokens)
        aa_position = type_ids.eq(self.aa_type) & output_masks
        struct_position = type_ids.eq(self.struct_type) & output_masks
        indices_aa = torch.where(aa_position)
        indices_struct = torch.where(struct_position)

        # HACK: all amino acid token id < 33, while all struct token id >= 33
        logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
        logits[indices_struct[0], indices_struct[1], :33] = -math.inf

        logits[..., self.special_token_list] = -math.inf
        
        # # logits = top_k_top_p_filtering(logits, top_p=0.95)

        # if sampling_strategy == 'vanilla':
        #     _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        # elif sampling_strategy == 'argmax':
        #     _scores, _tokens = logits.max(-1)
        # elif sampling_strategy == 'gumbel_argmax':
        #     noise_scale = 1.0
        #     _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)

        #     # 针对batch中的seq，如果一条seq的某aa类型频率过高就mask这条seq的对应aa类型的位置，重新加上功能条件sample一次
        #     self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)
        # else:
        #     raise NotImplementedError
        
        # output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        # output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # history.append(output_tokens.clone())

        # return dict(
        #     output_tokens=output_tokens,
        #     output_scores=output_scores,
        #     attentions=attentions, # [B, L, H, T, T]
        #     step=step + 1,
        #     max_step=max_step,
        #     history=history,
        #     hidden_states=net_out['last_hidden_state']
        # )

        logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "argmax":
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            noise_scale = temperature
            # TODO：结尾有logits的logmax
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )

            self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label, motif_struct_emb=motif_struct_emb)

            _tokens.masked_scatter_(
                ~output_masks, output_tokens[~output_masks]
            )

            
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )

            self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label, motif_struct_emb=motif_struct_emb)
        else:
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out["last_hidden_state"],
        )

    # def _reparam_decoding(
    #     self,
    #     output_tokens,
    #     output_scores,
    #     cur_tokens,
    #     cur_scores,
    #     decoding_strategy,
    #     xt_neq_x0,
    #     non_special_sym_mask,
    #     t,
    #     max_step,
    #     noise,
    # ):
    #     """
    #         This function is used to perform reparameterized decoding.
    #     """
    #     # output_tokens: [B, N]
    #     # output_scores: [B, N]
    #     # cur_tokens: [B, N]
    #     # cur_scores: [B, N]
    #     # xt_neq_x0: equivalent to not_b_t [B, N]
    #     # non_special_sym_mask: [B, N]
    #     # noise: either [B, N] or scalar (if using the mask noise)

    #     # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
    #     _, condition, topk_mode, schedule = decoding_strategy.split("-")

    #     # first set the denoising rate according to the schedule
    #     if schedule == "linear":
    #         rate = 1 - t / max_step
    #     elif schedule == "cosine":
    #         rate = np.cos(t / max_step * np.pi * 0.5)
    #     else:
    #         raise NotImplementedError

    #     # compute the cutoff length for denoising top-k positions
    #     cutoff_len = (
    #         non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
    #     ).long()
    #     # set the scores of special symbols to a large value so that they will never be selected
    #     _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
    #     to_be_resample = []
    #     for i, seq in enumerate(cur_tokens):
    #         most_token_dict = {}
    #         most_token = None
    #         most_token_num = -1
    #         for j, token in enumerate(seq):
    #             token = int(token)
    #             if token == self.pad_id:
    #                 continue
    #             if token not in most_token_dict:
    #                 most_token_dict[token] = [j]
    #             else:
    #                 most_token_dict[token].append(j)
    #             if len(most_token_dict[token]) > most_token_num:
    #                 most_token = token
    #                 most_token_num = len(most_token_dict[token])
    #         if most_token_num > len(seq) * 0.25:
    #             to_be_resample.append(i)
                
    #     # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
    #     if topk_mode.startswith("stochastic"):
    #         noise_scale = float(topk_mode.replace("stochastic", ""))
    #         lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    #     elif topk_mode == "deterministic":
    #         lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    #         if len(to_be_resample) > 0:
    #             noise_scale = 1.5
    #             #print(lowest_k_mask[to_be_resample[0]])
    #             lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
    #                                                          stochastic=True, temp=noise_scale * rate)
    #     else:
    #         raise NotImplementedError

    #     # Various choices to generate v_t := [v1_t, v2_t].
    #     # Note that
    #     #   v1_t governs the outcomes of tokens where b_t = 1,
    #     #   v2_t governs the outcomes of tokens where b_t = 0.

    #     # #### the `uncond` mode ####
    #     # In our reparameterized decoding,
    #     # both v1_t and v2_t can be fully determined by the current token scores .

    #     # #### the `cond` mode ####
    #     # However, we can also impose some conditional constraints on v1_t so that
    #     # the decoding can be performed in a more conservative manner.
    #     # For example, we can set v1_t = 0 only when
    #     # (the newly output tokens are the same as previous denoised results, AND
    #     # the current token score becomes lower, AND
    #     # the current token score is not in the top-k share among all tokens).
    #     if condition == "cond":
    #         not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
    #     elif condition == "uncond":
    #         not_v1_t = lowest_k_mask
    #     else:
    #         raise NotImplementedError

    #     # for b_t = 0, the token is set to noise if it is in the lowest k scores.
    #     not_v2_t = lowest_k_mask

    #     last_mask_position = xt_neq_x0
    #     masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
    #     if isinstance(noise, torch.Tensor):
    #         output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
    #     elif isinstance(noise, (int, float)):
    #         output_tokens.masked_fill_(masked_to_noise, noise)
    #     else:
    #         raise NotImplementedError("noise should be either a tensor or a scalar")
    #     output_scores.masked_fill_(masked_to_noise, -math.inf)

    #     masked_to_x0 = xt_neq_x0 & ~not_v2_t
    #     output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
    #     output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
    #     assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
    #     # b_{t} = (b_{t+1} & u_t) | v_t
    #     # For convenience, save the NOT of b_t for the next iteration
    #     # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
    #     #
    #     # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
    #     new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
    #     assert (new_xt_neq_x0 == not_v2_t).all()
    #     return new_xt_neq_x0, output_tokens, output_scores

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        type_ids,
        non_special_sym_mask,
        t,
        max_step,
        use_struct_only=False,
    ):
        def _reparam_process(
            output_tokens,
            output_scores,
            cur_tokens,
            cur_scores,
            xt_neq_x0,
            noise,
            non_special_sym_mask,
            is_all_mask=False,
        ):
            """This function is used to perform reparameterized decoding.

            output_tokens: [B, N]
            output_scores: [B, N]
            cur_tokens: [B, N]
            cur_scores: [B, N]
            xt_neq_x0: equivalent to not_b_t [B, N]
            non_special_sym_mask: [B, N]
            noise: either [B, N] or scalar (if using the mask noise)
            """

            # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
            _, condition, topk_mode, schedule = decoding_strategy.split("-")

            # first set the denoising rate according to the schedule
            if schedule == "linear":
                rate = 1 - t / max_step
            elif schedule == "cosine":
                rate = np.cos(t / max_step * np.pi * 0.5)
            else:
                raise NotImplementedError

            if is_all_mask:
                rate = 1

            # compute the cutoff length for denoising top-k positions
            cutoff_len = (
                non_special_sym_mask.sum(1, keepdim=True).type_as(
                    output_scores
                )
                * rate
            ).long()
            # set the scores of special symbols to a large value so that they will never be selected
            _scores_for_topk = cur_scores.masked_fill(
                ~non_special_sym_mask, 1000.0
            )

            # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
            if topk_mode.startswith("stochastic"):
                noise_scale = float(topk_mode.replace("stochastic", ""))
                lowest_k_mask = topk_masking(
                    _scores_for_topk,
                    cutoff_len,
                    stochastic=True,
                    temp=noise_scale * rate,
                )
            elif topk_mode == "deterministic":
                lowest_k_mask = topk_masking(
                    _scores_for_topk, cutoff_len, stochastic=False
                )

            elif topk_mode == "positionprior":
                lowest_k_mask_1 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask_2 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask = lowest_k_mask_1 | lowest_k_mask_2
            else:
                raise NotImplementedError

            # Various choices to generate v_t := [v1_t, v2_t].
            # Note that
            #   v1_t governs the outcomes of tokens where b_t = 1,
            #   v2_t governs the outcomes of tokens where b_t = 0.

            # #### the `uncond` mode ####
            # In our reparameterized decoding,
            # both v1_t and v2_t can be fully determined by the current token scores .

            # #### the `cond` mode ####
            # However, we can also impose some conditional constraints on v1_t so that
            # the decoding can be performed in a more conservative manner.
            # For example, we can set v1_t = 0 only when
            # (the newly output tokens are the same as previous denoised results, AND
            # the current token score becomes lower, AND
            # the current token score is not in the top-k share among all tokens).
            if condition == "cond":
                not_v1_t = (
                    (cur_tokens == output_tokens)
                    & (cur_scores < output_scores)
                    & lowest_k_mask
                )
            elif condition == "uncond":
                not_v1_t = lowest_k_mask
            else:
                raise NotImplementedError

            # for b_t = 0, the token is set to noise if it is in the lowest k scores.
            not_v2_t = lowest_k_mask

            last_mask_position = xt_neq_x0

            masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
            if isinstance(noise, torch.Tensor):
                output_tokens.masked_scatter_(
                    masked_to_noise, noise[masked_to_noise]
                )
            elif isinstance(noise, (int, float)):
                output_tokens.masked_fill_(masked_to_noise, noise)
            else:
                raise NotImplementedError(
                    "noise should be either a tensor or a scalar"
                )
            output_scores.masked_fill_(masked_to_noise, -math.inf)

            masked_to_x0 = xt_neq_x0 & ~not_v2_t
            output_tokens.masked_scatter_(
                masked_to_x0, cur_tokens[masked_to_x0]
            )
            output_scores.masked_scatter_(
                masked_to_x0, cur_scores[masked_to_x0]
            )
            assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
            # b_{t} = (b_{t+1} & u_t) | v_t
            # For convenience, save the NOT of b_t for the next iteration
            # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
            #
            # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t (?)
            new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
            assert (new_xt_neq_x0 == not_v2_t).all()
            return new_xt_neq_x0, output_tokens, output_scores

        aa_position = type_ids.eq(self.aa_type) & non_special_sym_mask
        struct_position = type_ids.eq(self.struct_type) & non_special_sym_mask
        new_xt_neq_x0 = xt_neq_x0.clone()
        new_xt_neq_x0_aa = new_xt_neq_x0.fill_(False)
        new_xt_neq_x0_struct = new_xt_neq_x0.fill_(False)
        if aa_position.any():
            new_xt_neq_x0_aa, output_tokens, output_scores = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & aa_position,
                noise=self.aa_mask_id,
                non_special_sym_mask=aa_position,
                is_all_mask=use_struct_only,
            )
        if struct_position.any():
            (
                new_xt_neq_x0_struct,
                output_tokens,
                output_scores,
            ) = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & struct_position,
                noise=self.struct_mask_id,
                non_special_sym_mask=struct_position,
            )
        new_xt_neq_x0 = new_xt_neq_x0_aa | new_xt_neq_x0_struct
        return new_xt_neq_x0, output_tokens, output_scores

    def generate(
        self, 
        batch,
        max_iter=None, 
        temperature=1.0, 
        partial_masks=None,
        unmasking_strategy="stochastic1.0",
        sampling_strategy='gumbel_argmax',
        use_struct_only=False,
    ):
        # tokenizer = tokenizer
        # max_iter = max_iter
        # temperature = temperature
        self.eval()
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens, where partial_masks will fix motif
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch.get("input_ids"), encoder_out=encoder_out, partial_masks=partial_masks)  #
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            type_ids=self.get_modality_type(initial_output_tokens),
        )

        prev_decoder_out['output_masks'] = self.get_non_special_symbol_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
        
        history_detail = []
        last_mask = prev_decoder_out["output_masks"].clone()
        
        for step in tqdm(range(max_iter), desc='Decoding'):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    go_label=batch.get('go_label', None),
                    ipr_label=batch.get('ipr_label', None),
                    seq_cond=batch.get('seq_cond', None),
                    ec_label=batch.get('ec_label', None),
                    motif_struct_emb=batch.get('motif_struct_emb', None),

                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                type_ids=prev_decoder_out["type_ids"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                use_struct_only=use_struct_only,
            )

            demask_pos = ((last_mask == 1) & (output_masks == 0)).nonzero(as_tuple=True)
            remask_pos = ((last_mask == 0) & (output_masks == 1)).nonzero(as_tuple=True)
            history_detail.append({
                "step": step + 1,
                "tokens": output_tokens.cpu(),
                "scores": output_scores.cpu(),
                "mask": output_masks.cpu(),
                "demask_pos": [x for x in zip(*[d.cpu().tolist() for d in demask_pos])],
                "remask_pos": [x for x in zip(*[d.cpu().tolist() for d in remask_pos])],
                "pred_tokens": decoder_out["output_tokens"],
            })
            last_mask = output_masks.clone()

        
            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            # print(f"step: {step}")
            # print(output_tokens)

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], decoder_out['output_scores'], history_detail
