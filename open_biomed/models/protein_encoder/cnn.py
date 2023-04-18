import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
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