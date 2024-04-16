import torch
import torch.nn as nn

from open_biomed.models.base_models import CellEncoder

class DeepCDR(torch.nn.Module):
    def __init__(self, input_dim, output_dim=100, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, output_dim)

    def forward(self, gexpr_input):
        # print(gexpr_input)
        gexpr_input = gexpr_input.float()
        x_gexpr = self.linear1(gexpr_input)
        x_gexpr = torch.tanh(x_gexpr)
        x_gexpr = self.norm(x_gexpr)
        x_gexpr = self.dropout(x_gexpr)
        x_gexpr = self.linear2(x_gexpr)
        return x_gexpr    

    def encode_cell(self, cell):
        return self.forward(cell)