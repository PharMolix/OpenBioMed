import torch
import torch.nn as nn
import torch.nn.functional as F

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.gnn_graphcl import GNNGraphCL
from open_biomed.models.text.base_transformers import BaseTransformers

class BioMedGPTCLIP(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(BioMedGPTCLIP, self).__init__()
        self.graph_encoder = GNNGraphCL(
            num_layer=config["structure"]["gin_num_layers"],
            emb_dim=config["structure"]["gin_hidden_dim"],
            gnn_type='gin',
            drop_ratio=config["structure"]["dropout"],
            JK='last',
        )
        self.text_encoder = BaseTransformers(config["text"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.main_model.h[-16:].parameters():
            param.requires_grad = True

        self.graph_proj_head = nn.Linear(self.graph_encoder.output_dim, config["projection_dim"])
        self.text_proj_head = nn.Linear(self.text_encoder.output_dim, config["projection_dim"])

    def forward(self, features_graph, features_text):
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def encode_mol(self, structure, proj=True, return_node_feats=False):
        h, node_feats = self.graph_encoder(structure)
        if proj:
            h = self.graph_proj_head(h)
            node_feats = self.graph_proj_head(node_feats)
        if return_node_feats:
            return h, node_feats
        else:
            return h

    def encode_structure_with_prob(self, structure, x, atomic_num_list, device):
        h, _ = self.graph_encoder(structure, x, atomic_num_list, device)
        return self.graph_proj_head(h) 

    def encode_text(self, text, proj=True):
        h = self.text_encoder(text)
        if proj:
            h = self.text_proj_head(h)
        return h