import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from open_biomed.models.base_models import ProteinEncoder

class ProtCNN(ProteinEncoder):
    def __init__(self, config):
        super(ProtCNN, self).__init__()
        self.output_dim = config["output_dim"]
        
        layer_size = len(config["in_ch"]) - 1
        self.conv = nn.ModuleList(
            [nn.Conv1d(
                in_channels = config["in_ch"][i], 
                out_channels = config["in_ch"][i + 1], 
                kernel_size = config["kernels"][i]
            ) for i in range(layer_size)]
        )
        self.conv = self.conv.double()
        hidden_dim = self._get_conv_output((config["vocab_size"], config["max_length"]))
        self.fc1 = nn.Linear(hidden_dim, config["output_dim"])

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x
    
    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

    def encode_protein(self, prot):
        return self.forward(prot)

class CNNGRU(ProteinEncoder):
    def __init__(self, config):
        super(CNNGRU, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=config["input_dim"], out_channels=config["cnn_dim"], kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(config["cnn_dim"])
        self.biGRU = nn.GRU(config["cnn_dim"], config["cnn_dim"], bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(config["pool_size"], stride=config["pool_size"])
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(config["input_len"] / config["pool_size"]), config["output_dim"])

    def forward(self, prot):
        x = prot.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x, _ = self.biGRU(x)
        x = self.global_avgpool1d(x)
        x = x.squeeze()
        x = self.fc1(x)

        return x

    def encode_protein(self, prot):
        return self.forward(prot)

class ResConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, pool_size):
        super(ResConvGRU, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.biGRU = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, prot):
        x = prot.transpose(1, 2)
        x = self.conv1d(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        h, _ = self.biGRU(x)
        x = torch.cat([h, x], dim=2)
        return x

class CNNPIPR(ProteinEncoder):
    def __init__(self, config):
        super(CNNPIPR, self).__init__()
        self.convs = nn.Sequential(
            ResConvGRU(config["input_dim"], config["hidden_dim"], 2),
            ResConvGRU(3 * config["hidden_dim"], config["hidden_dim"], 2),
            ResConvGRU(3 * config["hidden_dim"], config["hidden_dim"], 2),
        )
        self.last_conv = nn.Conv1d(in_channels=config["hidden_dim"] * 3, out_channels=config["hidden_dim"], kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config["input_len"] // 8, config["output_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        self.act = nn.LeakyReLU(0.3)
        self.output_dim = config["output_dim"]

    def forward(self, prot):
        x = self.convs(prot)
        x = x.transpose(1, 2)
        x = self.last_conv(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc(x)
        return self.act(x)

    def encode_protein(self, prot):
        return self.forward(prot)