import json

import torch
import torch.nn as nn

from open_biomed.models import SUPPORTED_PROTEIN_ENCODER, SUPPORTED_KNOWLEDGE_ENCODER
from open_biomed.models.predictor import MLP

class PPISeqModel(nn.Module):
    def __init__(self, config, num_classes):
        super(PPISeqModel, self).__init__()
        protein_encoder_config = json.load(open(config["encoder"]["config_path"], "r"))
        self.protein_encoder = SUPPORTED_PROTEIN_ENCODER[config["encoder"]["name"]](protein_encoder_config)
        self.feature_fusion = config["feature_fusion"]
        if self.feature_fusion == 'concat':
            in_dim = self.protein_encoder.output_dim * 2
        else:
            in_dim = self.protein_encoder.output_dim
        self.pred_head = MLP(config["pred_head"], in_dim, num_classes)

    def forward(self, prot1, prot2):
        x1 = self.protein_encoder(prot1)
        x2 = self.protein_encoder(prot2)
        #print(x1, x2)
        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        return self.pred_head(x)

class PPIGraphModel(nn.Module):
    def __init__(self, config, num_classes):
        super(PPIGraphModel, self).__init__()
        self.graph_encoder = SUPPORTED_KNOWLEDGE_ENCODER[config["name"]](config)
        self.feature_fusion = config["feature_fusion"]
        if self.feature_fusion == 'concat':
            in_dim = self.graph_encoder.output_dim * 2
        else:
            in_dim = self.graph_encoder.output_dim
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, prot1, prot2, graph):
        x = self.graph_encoder(graph)
        x1, x2 = x[prot1], x[prot2]
        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc(x)

        return x

class DeepEIK4PPI(nn.Module):
    def __init__(self, config, num_classes):
        super(DeepEIK4PPI, self).__init__()

    def forward(self, prot1, prot2):
        pass