import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.gnn_graphmvp import GNNGraphMVP
from open_biomed.models.multimodal.molfm.modeling_t5 import T5Config, T5ForConditionalGeneration

class MolFMPlus(MolEncoder, TextEncoder):
    def __init__(self, config):
        super().__init__()
        self.graph_config = config["structure"]
        self.max_n_atoms = config["max_n_atoms"]
        self.projection_dim = config["projection_dim"]
        self.tokenizer = T5Tokenizer.from_pretrained(config["tokenizer"])
        t5_config = T5Config.from_json_file(config["text"]["config_file"])
        
        self.graph_model = GNNGraphMVP(
            num_layer=self.graph_config["gin_num_layers"],
            emb_dim=self.graph_config["gin_hidden_dim"],
            gnn_type="gin",
            drop_ratio=self.graph_config["drop_ratio"],
            JK="last",
        )
        if "ckpt" in self.graph_config:
            print("load graph checkpoint from ", self.graph_config["ckpt"])
            self.graph_model.load_state_dict(torch.load(self.graph_config["ckpt"], map_location="cpu"), strict=False)
        self.structure_proj_head = nn.Linear(t5_config.hidden_size, self.projection_dim)
        #self.structure_proj_head = nn.Linear(self.graph_model.output_dim, self.projection_dim)
        self.graph_linear = nn.Linear(self.graph_model.output_dim, t5_config.hidden_size)
        
        self.text_model = T5ForConditionalGeneration(t5_config)
        if "ckpt" in config["text"]:
            ckpt = torch.load(config["text"]["ckpt"], map_location="cpu").state_dict()
            missing_keys, unexpected_keys = self.text_model.load_state_dict(ckpt, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        self.text_proj_head = nn.Linear(t5_config.hidden_size, self.projection_dim)
        self.h_proj = nn.Linear(t5_config.hidden_size, self.projection_dim)
        self.t_proj = nn.Linear(t5_config.hidden_size, self.projection_dim)
        self.mtm_head = nn.Linear(t5_config.hidden_size, 2)

    def forward(self, mol=None, text=None, labels=None):
        assert (mol is not None or text is not None) and labels is not None
        if mol is not None:
            _, encoder_hidden_states = self.encode_mol(mol, proj=False, return_node_feats=True)
            encoder_attention_mask = mol["smi"]["attention_mask"]
        elif text is not None:
            _, encoder_hidden_states = self.encode_text(text, proj=False, output_hidden_states=True)
            encoder_attention_mask = text["attention_mask"]
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states,
            hidden_states=None,
            attentions=None
        )
        decoder_attention_mask = labels["attention_mask"]
        labels = labels["input_ids"].masked_fill(~labels["attention_mask"].bool(), -100)
        return self.text_model.decoder(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def get_graph_feats(self, graph, batch_size):
        graph_embeds, node_embeds = self.graph_model(graph)
        all_node_feats = self.graph_linear(node_embeds)
        # serialize node feature
        node_feats = []
        node_attention_mask = []
        for i in range(batch_size):
            feat = all_node_feats[torch.where(graph.batch == i)]
            if feat.shape[0] < self.max_n_atoms:
                node_feats.append(torch.cat((
                    feat,
                    torch.zeros(self.max_n_atoms - feat.shape[0], feat.shape[1]).to(feat.device)
                ), dim=0))
                node_attention_mask.append(torch.cat((
                    torch.ones(feat.shape[0]).to(feat.device), 
                    torch.zeros(self.max_n_atoms - feat.shape[0]).to(feat.device)
                ), dim=0))
            else:
                node_feats.append(feat[:self.max_n_atoms, :])
                node_attention_mask.append(torch.ones(self.max_n_atoms).to(feat.device))
        node_feats = torch.stack(node_feats, dim=0)
        node_attention_mask = torch.stack(node_attention_mask, dim=0)
        return graph_embeds, node_feats, node_attention_mask

    def seq_wrap(self, seq1, seq2):
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

    def encode_mol(self, mol, proj=False, return_node_feats=False):
        graph, smi = mol["graph"], mol["smiles"]
        batch_size = smi["input_ids"].shape[0]
        _, node_embeds, node_attention_mask = self.get_graph_feats(graph, batch_size)
        mol_embeds = self.text_model.encoder(
            input_ids=smi["input_ids"],
            attention_mask=smi["attention_mask"],
            #encoder_hidden_states=node_embeds,
            #encoder_attention_mask=node_attention_mask,
            return_dict=True
        ).last_hidden_state
        glob = mol_embeds[:, 0, :]
        if proj:
            glob = self.structure_proj_head(glob)
        if not return_node_feats:
            return glob
        else:
            return glob, mol_embeds

    def encode_text(self, text, proj=False, output_hidden_states=False):
        text_embeds = self.text_model.encoder(text["input_ids"], attention_mask=text["attention_mask"], return_dict=True).last_hidden_state
        glob = text_embeds[:, 0, :]
        if proj:
            glob = self.text_proj_head(text_embeds)
        if not output_hidden_states:
            return glob
        else:
            return glob, text_embeds

    def predict_similarity_score(self, data):
        preds = self.encode_multimodal(data["structure"], data["text"]).last_hidden_state[:, 0, :]
        return F.softmax(self.mtm_head(preds), dim=-1)[:, 1]

    def encode_multimodal(self, mol, text):
        batch_size = text["input_ids"].shape[0]
        graph, smi = mol["graph"], mol["smiles"]
        _, node_embeds, node_attention_mask = self.get_graph_feats(graph, batch_size)
        wrapped_input_ids, wrapped_attention_mask = self.seq_wrap(smi, text)

        return self.text_model.encoder(
            wrapped_input_ids,
            attention_mask=wrapped_attention_mask,
            encoder_hidden_states=node_embeds,
            encoder_attention_mask=node_attention_mask,
            return_dict=True
        )

    def decode_text(self, mol=None, text=None, num_beams=5, max_length=512):
        assert mol is not None or text is not None
        if mol is not None:
            _, encoder_hidden_states = self.encode_mol(mol, proj=False, return_node_feats=True)
            encoder_attention_mask = mol["smi"]["attention_mask"]
        elif text is not None:
            _, encoder_hidden_states = self.encode_text(text, proj=False, output_hidden_states=True)
            encoder_attention_mask = text["attention_mask"]
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states,
            hidden_states=None,
            attentions=None
        )
        return self.text_model.decoder.generate(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )