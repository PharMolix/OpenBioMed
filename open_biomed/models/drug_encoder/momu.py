import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel

from models.drug_encoder.momu_gnn import MoMuGNN

class TextEncoder(nn.Module):
    def __init__(self, pretrained=True, model_name_or_path=None, dropout=0.0):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            self.main_model = BertModel.from_pretrained(model_name_or_path)
        else:
            config = BertConfig(vocab_size=31090, )
            self.main_model = BertModel(config)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        output = self.main_model(**text)["pooler_output"]
        logits = self.dropout(output)
        return logits

class MoMu(nn.Module):
    def __init__(self, config):
        super(MoMu, self).__init__()

        self.gin_hidden_dim = config["gin_hidden_dim"]
        self.gin_num_layers = config["gin_num_layers"]
        self.drop_ratio = config["drop_ratio"]
        self.graph_pooling = config["graph_pooling"]
        self.graph_self = config["graph_self"]

        self.bert_dropout = config["bert_dropout"]
        self.bert_hidden_dim = config["bert_hidden_dim"]

        self.projection_dim = config["projection_dim"]

        self.graph_encoder = MoMuGNN(
            num_layer=self.gin_num_layers,
            emb_dim=self.gin_hidden_dim,
            gnn_type='gin',
            drop_ratio=self.drop_ratio,
            JK='last',
        )

        self.text_encoder = TextEncoder(pretrained=False, dropout=self.bert_dropout)

        self.graph_proj_head = nn.Sequential(
            nn.Linear(self.gin_hidden_dim, self.gin_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.gin_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )
        self.output_dim = self.projection_dim
        # self.output_dim = self.gin_hidden_dim

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

    def encode_structure(self, structure, proj=True):
        h, _ = self.graph_encoder(structure)
        if proj:
            h = self.graph_proj_head(h)
        return h

    def encode_structure_with_prob(self, structure, x, atomic_num_list, device):
        h, _ = self.graph_encoder(structure, x, atomic_num_list, device)
        return self.graph_proj_head(h) 

    def encode_text(self, text):
        h = self.text_encoder(text)
        return self.text_proj_head(h)