import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.unimol import UniMol
from open_biomed.models.multimodal.molkformer.kformer import BertConfig, BertLMHeadModel
from open_biomed.utils.mol_utils import get_biot5_tokenizer, get_unimol_dictionary

class MVMol(MolEncoder, TextEncoder):
    def __init__(self, config):
        super().__init__()
        self.structure_config = config["structure"]
        self.qformer_config = BertConfig.from_json_file(config["qformer_config_file"])
        self.projection_dim = config["projection_dim"]
        self.max_n_atoms = config["max_n_atoms"]
        self.num_query_tokens = self.qformer_config.num_query_tokens
        self.encoder_tokenizer = BertTokenizer.from_pretrained(config["encoder_tokenizer"])
        self.decoder_tokenizer = get_biot5_tokenizer({
            "model_name_or_path": config["decoder_tokenizer"],
            "path_selfies": config["path_selfies"],
            "max_length": 512
        })

        self.structure_encoder = UniMol(
            json.load(open(self.structure_config["config_path"], "r")),
            get_unimol_dictionary(self.structure_config["dict_path"]),
        )
        if "ckpt" in self.structure_config:
            self.structure_encoder.load_state_dict(torch.load(self.structure_config["ckpt"], map_location="cpu"), strict=False)
        self.structure_linear = nn.Linear(self.structure_encoder.output_dim, self.qformer_config.hidden_size)
        self.structure_proj_head = nn.Linear(self.qformer_config.hidden_size, self.projection_dim)

        self.qformer = BertLMHeadModel(self.qformer_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, self.qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=self.qformer_config.initializer_range)
        self.text_proj_head = nn.Linear(self.qformer_config.hidden_size, self.projection_dim)
        self.mtm_head = nn.Linear(self.qformer_config.hidden_size, 2)

        decoder_config = T5Config.from_json_file(config["decoder"]["config_file"])
        self.text_decoder = T5ForConditionalGeneration(decoder_config)
        self.text_decoder.resize_token_embeddings(35073)
        self.enc2dec = nn.Linear(self.qformer_config.hidden_size, self.text_decoder.config.hidden_size)
        
        self.norm = True

    def _get_structure_features(self, mol):
        node_embeds, node_attention_mask = self.structure_encoder(*mol)
        node_embeds = self.structure_linear(node_embeds)
        if node_attention_mask is not None:
            node_attention_mask = 1 - node_attention_mask.long()
        else:
            node_attention_mask = torch.ones(node_embeds.shape[:-1], dtype=torch.long).to(node_embeds.device)
        return node_embeds, node_attention_mask

    def forward(self, mol, text=None, prompt=None):
        # calculate molecule feature
        batch_size = mol[0].shape[0]
        node_embeds, node_attention_mask = self._get_structure_features(mol)
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        if text is None and prompt is None:
            input_ids = None
            attention_mask = torch.ones(query_embeds.shape[:-1], dtype=torch.long).to(query_embeds.device)
        else:
            input_ids, attention_mask = self.seq_wrap(prompt, text)
            attention_mask = torch.cat([torch.ones(query_embeds.shape[:-1], dtype=torch.long).to(query_embeds.device), attention_mask], dim=1)

        return self.qformer.bert(
            input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=node_embeds,
            encoder_attention_mask=node_attention_mask,
            return_dict=True,
        ).last_hidden_state[:, :self.num_query_tokens, :]

    def seq_wrap(self, seq1, seq2):
        if seq1 is None:
            return seq2["input_ids"], seq2["attention_mask"]
        if seq2 is None:
            return seq1["input_ids"], seq1["attention_mask"]
        batch_size = seq1["input_ids"].shape[0]
        wrapped_inputs, wrapped_attention_mask = [], []
        for i in range(batch_size):
            cur_len = seq1["attention_mask"][i].sum()
            wrapped_inputs.append(torch.cat([
                seq1["input_ids"][i, :cur_len],
                seq2["input_ids"][i],
                seq1["input_ids"][i, cur_len:]
            ], dim=0))
            wrapped_attention_mask.append(torch.cat([
                seq1["attention_mask"][i, :cur_len],
                seq2["attention_mask"][i],
                seq1["attention_mask"][i, cur_len:]
            ], dim=0))
        return torch.stack(wrapped_inputs, dim=0), torch.stack(wrapped_attention_mask, dim=0)

    def feat_wrap(self, seq1_feats, seq1_attn, seq2_feats, seq2_attn):
        batch_size = seq1_feats.shape[0]
        wrapped_feats, wrapped_attention_mask = [], []
        for i in range(batch_size):
            cur_len = seq1_attn[i].sum()
            wrapped_feats.append(torch.cat([
                seq1_feats[i, :cur_len],
                seq2_feats[i],
                seq1_feats[i, cur_len:]
            ], dim=0))
            wrapped_attention_mask.append(torch.cat([
                seq1_attn[i, :cur_len],
                seq2_attn[i],
                seq1_attn[i, cur_len:]
            ], dim=0))
        return torch.stack(wrapped_feats, dim=0), torch.stack(wrapped_attention_mask, dim=0)

    def encode_mol(self, mol, proj=False):
        if "structure" in mol:
            s_inp = mol["structure"]
        else:
            s_inp = mol
        if "conformation" in s_inp:
            s_inp = s_inp["conformation"]
        # print(mol["text"])
        mol_embeds = self.forward(s_inp, prompt=mol["text"] if "text" in mol else None)
        # mol_embeds = self.forward(s_inp)
        if proj:
            mol_embeds = F.normalize(self.structure_proj_head(mol_embeds), dim=-1)
        return mol_embeds

    def encode_text(self, text, return_cls=True, proj=False):
        text_embeds = self.qformer.bert(
            text["input_ids"], 
            attention_mask=text["attention_mask"],
            return_dict=True,
        ).last_hidden_state
        if return_cls:
            text_embeds = text_embeds[:, 0, :]
        if proj:
            text_embeds = F.normalize(self.text_proj_head(text_embeds), dim=-1)
        return text_embeds

    def decode(self, mol, num_beams, max_length):
        h_graph = self.encode_mol(mol)
        h_graph = self.enc2dec(h_graph)
        #h_smi = self.text_decoder.encoder(**mol["structure"]["SMILES"]).last_hidden_state
        #h = torch.cat([h_graph, h_smi], dim=1)
        h = h_graph
        attention_mask = torch.ones(h_graph.shape[:-1], dtype=torch.long).to(h.device)
        #attention_mask = torch.cat([attention_mask, mol["structure"]["SMILES"].attention_mask], dim=1)

        """
        h_coord, mask_coord = self._get_structure_features(mol["structure"]["conformation"])
        h_smi = self.text_decoder.encoder(**mol["structure"]["SMILES"]).last_hidden_state
        h, attention_mask = self.feat_wrap(h_coord, mask_coord, h_smi, mol["structure"]["SMILES"].attention_mask)
        """
        h = BaseModelOutput(
            last_hidden_state=h,
            hidden_states=None,
            attentions=None
        )
        outputs = self.text_decoder.generate(
            encoder_outputs=h,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )
        return outputs
        #return self.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def decode_mol(self, text, num_beams, max_length):
        h_text = self.text_decoder.encoder(**text).last_hidden_state
        h = BaseModelOutput(
            last_hidden_state=h_text,
            hidden_states=None,
            attentions=None
        )
        return self.text_decoder.generate(
            encoder_outputs=h,
            attention_mask=text["attention_mask"],
            num_beams=num_beams,
            max_length=max_length
        )

    def predict_similarity_score(self, mol, text):
        if "text" in mol:
            prompt = mol["text"]
            mol = mol["structure"]
        else:
            prompt = None
            mol = mol["structure"]
        preds = self.forward(mol, text, prompt=prompt)
        return F.softmax(self.mtm_head(preds).mean(dim=1), dim=-1)[:, 1]

    def cmm_loss(self, mol, text, prompt):
        batch_size = mol['atoms'].shape[0]
        node_embeds, node_attention_mask = self._get_structure_features(mol)
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        if prompt is not None:
            prompt_embeds = self.qformer.get_input_embeddings()(prompt["input_ids"])
            query_attention_mask = torch.cat([torch.ones(query_embeds.shape[:-1], dtype=torch.long).to(self.device), prompt["attention_mask"]], dim=1)
        else:
            prompt_embeds = None
            query_attention_mask = torch.ones(query_embeds.shape[:-1], dtype=torch.long).to(self.device)
        query_outputs = self.qformer.bert(
            encoder_embeds=prompt_embeds,
            query_embeds=query_embeds,
            attention_mask=query_attention_mask,
            encoder_hidden_states=node_embeds,
            encoder_attention_mask=node_attention_mask,
            return_dict=True
        ).last_hidden_state[:, :self.num_query_tokens, :]
        mol_feats = F.normalize(self.structure_proj_head(query_outputs), dim=-1)

        text_outputs = self.qformer.bert(
            text["input_ids"], 
            attention_mask=text["attention_mask"], 
            return_dict=True
        )
        text_embeds = text_outputs.last_hidden_state
        text_feats = F.normalize(self.text_proj_head(text_embeds[:, 0, :]), dim=-1)

        sim_m2t = torch.matmul(mol_feats.unsqueeze(1), text_feats.unsqueeze(-1)).squeeze()
        sim_m2t, _ = sim_m2t.max(dim=-1) 
        sim_t2m = torch.matmul(text_feats.unsqueeze(1).unsqueeze(1), mol_feats.transpose(1, 2)).squeeze()
        sim_t2m, _ = sim_t2m.max(dim=-1)
        
        # find hard negatives
        with torch.no_grad():
            weights_m2t = F.softmax(sim_m2t, dim=1) + 1e-4
            weights_m2t.fill_diagonal_(0.0)
            weights_t2m = F.softmax(sim_t2m, dim=1) + 1e-4
            weights_t2m.fill_diagonal_(0.0)
        idx_neg_m2t = []
        for i in range(batch_size):
            idx_neg_m2t.append(torch.multinomial(weights_m2t[i], 1).item())
        idx_neg_m2t = torch.tensor(idx_neg_m2t, dtype=int).to(node_embeds)
        idx_neg_t2m = []
        for i in range(batch_size):
            idx_neg_t2m.append(torch.multinomial(weights_t2m[i], 1).item())

        node_embeds_mtm = torch.cat([node_embeds, node_embeds, node_embeds[idx_neg_t2m]], dim=0)
        node_attention_mask_mtm = torch.cat([node_attention_mask, node_attention_mask, node_attention_mask[idx_neg_t2m]], dim=0)
        wrapped_input_ids, wrapped_attention_mask = self.seq_wrap(prompt, text)
        text_input_ids_mtm = torch.cat([wrapped_input_ids, wrapped_input_ids[idx_neg_m2t], wrapped_input_ids], dim=0)
        text_attention_mask_mtm = torch.cat([wrapped_attention_mask, wrapped_attention_mask[idx_neg_m2t], wrapped_attention_mask], dim=0)
        query_embeds_mtm = self.query_tokens.expand(node_embeds_mtm.shape[0], -1, -1)
        query_attention_mask_mtm = torch.ones(query_embeds_mtm.shape[:-1], dtype=torch.long).to(query_embeds_mtm.device)
        text_attention_mask_mtm = torch.cat([query_attention_mask_mtm, text_attention_mask_mtm], dim=1)
        mtm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)], dim=0).to(query_embeds_mtm.device)

        output = self.qformer.bert(
            input_ids=text_input_ids_mtm,
            query_embeds=query_embeds_mtm,
            attention_mask=text_attention_mask_mtm,
            encoder_hidden_states=node_embeds_mtm,
            encoder_attention_mask=node_attention_mask_mtm,
            return_dict=True
        )
        mtm_output = self.mtm_head(output["last_hidden_state"][:, : self.num_query_tokens, :]).mean(dim=1)
        loss_mtm = F.cross_entropy(mtm_output, mtm_labels)
        return loss_mtm

    def causal_generation_loss(self, mol, text):
        labels = text["input_ids"].masked_fill(~text["attention_mask"].bool(), -100)
        """
        h_graph = self.encode_mol(mol)
        h_graph = self.enc2dec(h_graph)
        """
        h_coord, mask_coord = self._get_structure_features(mol["structure"]["conformation"])
        h_smi = self.text_decoder.encoder(**mol["structure"]["SMILES"]).last_hidden_state
        """
        h = torch.cat([h_graph, h_smi], dim=1)
        attention_mask = torch.ones(h_graph.shape[:-1], dtype=torch.long).to(h.device)
        attention_mask = torch.cat([attention_mask, mol["structure"]["SMILES"].attention_mask], dim=1)
        """
        h, attention_mask = self.feat_wrap(h_coord, mask_coord, h_smi, mol["structure"]["SMILES"].attention_mask)
        h = BaseModelOutput(
            last_hidden_state=h,
            hidden_states=None,
            attentions=None
        )
        return self.text_decoder(
            encoder_outputs=h,
            attention_mask=attention_mask,
            decoder_attention_mask=text["attention_mask"],
            return_dict=True,
            labels=labels
        ).loss

    def mol_generation_loss(self, text, mol):
        labels = mol["input_ids"].masked_fill(~mol["attention_mask"].bool(), -100)
        """
        h_text = self.encode_text(text, return_cls=False, proj=False)
        h_text = self.enc2dec(h_text)
        """
        h_text = self.text_decoder.encoder(**text).last_hidden_state
        h = BaseModelOutput(
            last_hidden_state=h_text,
            hidden_states=None,
            attentions=None
        )
        return self.text_decoder(
            encoder_outputs=h,
            attention_mask=text["attention_mask"],
            decoder_attention_mask=mol["attention_mask"],
            return_dict=True,
            labels=labels
        ).loss