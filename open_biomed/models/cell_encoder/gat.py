import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, max_pool


class CellGAT(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        # self.activations = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                conv = GATConv(self.num_feature, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # True or False
            # activation = nn.PReLU(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)
            # self.activations.append(activation)

    def forward(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return node_representation
    
    def grad_cam(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return cell_node, node_representation