import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

from torch_geometric.nn import GINConv, JumpingKnowledge

from open_biomed.models.base_models import KnowledgeEncoder
from open_biomed.models.protein.cnn import CNNGRU

SUPPORTED_FEATURE_NETWORK = {
    "cnn_gru": CNNGRU,
    "linear": lambda x: nn.Linear(x["input_dim"], x["output_dim"]),
}

class GIN(KnowledgeEncoder):
    def __init__(self, config):
        super(GIN, self).__init__()
        self.use_jk = config["gnn"]["use_jk"]
        self.train_eps = config["gnn"]["train_eps"]
        self.hidden_dim = config["gnn"]["hidden_dim"]

        self.fn = SUPPORTED_FEATURE_NETWORK[config["feature_network"]["name"]](config["feature_network"])

        self.gin_conv1 = GINConv( 
            nn.Sequential(
                nn.Linear(config["feature_network"]["output_dim"], self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(config["gnn"]["hidden_dim"]),
            ), train_eps=self.train_eps
        )
        self.gin_convs = torch.nn.ModuleList()
        for i in range(config["gnn"]["num_layers"] - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dim),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(config["gnn"]["num_layers"] * self.hidden_dim, self.hidden_dim)
        else:
            self.lin1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config["dropout"])
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_dim = self.hidden_dim
    
    def reset_parameters(self):
        self.fn.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()
        
        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index

        x = x.squeeze()
        x = self.fn(x)

        x = self.gin_conv1(x, edge_index)
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs += [x]

        if self.use_jk:
            x = self.jump(xs)
        
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        # x  = torch.add(x, x_)
        return x

    def encode_knowledge(self, kg):
        return self.forward(kg)
        