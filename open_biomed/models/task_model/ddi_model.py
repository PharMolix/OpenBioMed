import json
import random
import torch
import torch.nn as nn

from open_biomed.models import SUPPORTED_MOL_ENCODER
from open_biomed.models.predictor import MLP

class SparseAttention(nn.Module):
    def __init__(self, config, kge, mlp_dim):
        super(SparseAttention, self).__init__()
        self.config = config
        kge_id = list(kge.keys())
        kge_id.sort()
        kge_emb_list = [kge[k] for k in kge_id]
        self.kge_emb = torch.vstack(kge_emb_list)

        self.span_encoding = MLP(
            {
                'hidden_size': [mlp_dim],
                'dropout': 0.0,
                'activation': 'relu',
                'batch_norm': False
            },
            mlp_dim,
            self.kge_emb[0].shape[0]
        )

    def forward(self, x):
        spanned = self.span_encoding(x)
        self.kge_emb = self.kge_emb.to(spanned.dtype)
        dotprod = spanned @ self.kge_emb.T
        if len(dotprod.shape) == 1:
            dotprod = dotprod.unsqueeze(0)
        topmem, topmem_idx = torch.topk(dotprod, self.config['k'], dim=1)
        topmem = torch.exp(topmem)
        topmem = torch.nn.functional.softmax(topmem, dim=1)
        batch_size = topmem_idx.shape[0]
        topval = torch.stack([self.kge_emb[topmem_idx[i, :], :] for i in range(batch_size)])
        value = torch.vstack([topmem[i, :].unsqueeze(0) @ topval[i, :] for i in range(batch_size)])
        return value


class MultiHeadSparseAttention(nn.Module):
    def __init__(self, config, kge, mlp_dim):
        super(MultiHeadSparseAttention, self).__init__()
        n_heads = config['heads']
        kge_dim = kge[list(kge.keys())[0]].shape[0]
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(SparseAttention(config, kge, mlp_dim))
        self.linear = nn.Linear(n_heads * kge_dim, kge_dim)

    def forward(self, x):
        heads_output = [head(x) for head in self.heads]
        x = torch.hstack(heads_output)
        x = self.linear(x)
        return x


class DDIModel(nn.Module):
    def __init__(self, config, pred_dim):
        super(DDIModel, self).__init__()
        drug_encoder_config = json.load(open(config["drug"]["config_path"], "r"))
        self.drug_encoder = SUPPORTED_MOL_ENCODER[config["drug"]["name"]](drug_encoder_config)
        if "ckpt" in drug_encoder_config:
            state_dict = torch.load(drug_encoder_config["ckpt"], map_location="cpu")
            if "param_key" in state_dict:
                state_dict = state_dict[drug_encoder_config["param_key"]]
            self.drug_encoder.load_state_dict(state_dict)
            print("load ckpt from ", drug_encoder_config["ckpt"])
        self.pred_head = MLP(config["pred_head"], 2*self.drug_encoder.output_dim, pred_dim)

    def forward(self, drugA, drugB):
        h_drugA_structure = self.drug_encoder.encode_mol(drugA, proj=False, return_node_feats=False)
        h_drugB_structure = self.drug_encoder.encode_mol(drugB, proj=False, return_node_feats=False)
        h_structure = torch.cat((h_drugA_structure, h_drugB_structure), dim=1)
        return self.pred_head(h_structure)


class DeepEIK4DDI(nn.Module):
    def __init__(self, config, pred_dim):
        super(DeepEIK4DDI, self).__init__()
        self.config = config
        self.use_sparse_attention = config['sparse_attention']['active']
        self.projection_dim = config['projection_dim']
        self.kge = config['kge']
        self.kge_dim = self.kge[list(self.kge.keys())[0]].shape[0]

        if self.kge is None and self.use_sparse_attention:
            raise RuntimeError('No KGE to use for sparse attention')

        drug_encoder_config = json.load(open(config["drug"]["structure"]["config_path"], "r"))
        self.drug_structure_encoder = SUPPORTED_MOL_ENCODER[config["drug"]["structure"]["name"]](drug_encoder_config)
        if "init_checkpoint" in drug_encoder_config.keys():
            encoder_ckpt = drug_encoder_config["init_checkpoint"]
            assert encoder_ckpt
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            missing_keys, unexpected_keys = self.drug_structure_encoder.load_state_dict(ckpt, strict=False)
            print("encoder missing_keys: ", missing_keys)
            print("encoder unexpected_keys: ", unexpected_keys)

        if self.use_sparse_attention:
            self.mask_prob = config['sparse_attention']['mask_prob']
            # Only used when KGE is not available (all zeros)
            kge_drug = {k: self.kge[k] for k in self.kge if k[0] == 'D'}
            self.sparse_attn_config = config['sparse_attention']
            self.drug_sparse_attn = MultiHeadSparseAttention(self.sparse_attn_config,
                                                             kge_drug,
                                                             self.drug_structure_encoder.output_dim)

        self.structure_hidden_dim = 2 * self.drug_structure_encoder.output_dim

        self.kg_project = nn.Sequential(
            nn.Linear(2 * config["drug"]["kg"]["embedding_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        self.text_project = nn.Sequential(
            nn.Linear(config["text_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        self.pred_head = MLP(config["pred_head"], self.structure_hidden_dim + 2 * self.projection_dim, pred_dim)

    def forward(self, drugA, drugB):
        batch_size = drugA['kg'].shape[0]

        h_drugA_structure = self.drug_structure_encoder.encode_mol(drugA["structure"])
        if self.config['drug']['structure']['name'] == 'graphmvp':
            h_drugA_structure = h_drugA_structure[0]  # extracting h_graph from (h_graph, h_node)
        h_drugB_structure = self.drug_structure_encoder.encode_mol(drugB["structure"])
        if self.config['drug']['structure']['name'] == 'graphmvp':
            h_drugB_structure = h_drugB_structure[0]  # extracting h_graph from (h_graph, h_node)

        h_structure = torch.cat((h_drugA_structure, h_drugB_structure), dim=1)

        h_drugA_kg = drugA['kg']
        h_drugB_kg = drugB['kg']

        if self.use_sparse_attention:
            if self.training:
                for i in range(batch_size):
                    rand_drugA = random.uniform(0, 1)
                    rand_drugB = random.uniform(0, 1)
                    if rand_drugA < self.mask_prob:
                        h_drugA_kg[i, :] = 0
                    if rand_drugB < self.mask_prob:
                        h_drugB_kg[i, :] = 0

            drugA_nokge = torch.all(h_drugA_kg == 0, dim=1)
            drugA_nokge = torch.nonzero(drugA_nokge)
            if drugA_nokge.numel() > 0:
                drugA_nokge = torch.flatten(drugA_nokge)
                h_drugA_structure_subset = h_drugA_structure[drugA_nokge, :]
                if len(h_drugA_structure_subset.shape) == 1:
                    h_drugA_structure_subset = h_drugA_structure_subset.unsqueeze(0)
                h_drugA_nokge = self.drug_sparse_attn(h_drugA_structure_subset)
                for i, j in enumerate(drugA_nokge):
                    h_drugA_kg[j, :] = h_drugA_nokge[i, :]

            drugB_nokge = torch.all(h_drugB_kg == 0, dim=1)
            drugB_nokge = torch.nonzero(drugB_nokge)
            if drugB_nokge.numel() > 0:
                drugB_nokge = torch.flatten(drugB_nokge)
                h_drugB_structure_subset = h_drugB_structure[drugB_nokge, :]
                if len(h_drugB_structure_subset.shape) == 1:
                    h_drugB_structure_subset = h_drugB_structure_subset.unsqueeze(0)
                h_drugB_nokge = self.drug_sparse_attn(h_drugB_structure_subset)
                for i, j in enumerate(drugB_nokge):
                    h_drugB_kg[j, :] = h_drugB_nokge[i, :]

        h_kg = self.kg_project(torch.cat((h_drugA_kg, h_drugB_kg), dim=1))

        h_text = drugA["text"]
        h_text = self.text_project(h_text)

        h = torch.cat((h_structure, h_kg, h_text), dim=1)
        return self.pred_head(h)


SUPPORTED_DDI_NETWORKS = {
    'deepeik': DeepEIK4DDI,
    'graphmvp': DDIModel,
    'molfm': DDIModel
}