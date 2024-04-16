import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.utils import add_self_loops, softmax, degree

from open_biomed.models.base_models import MolEncoder, TextEncoder
from transformers import BertModel

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate([119, 4, 12, 12, 10, 6, 6, 2, 2]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate([5, 6, 2]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


class GNN_graphpred(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536 """

    def __init__(self, num_layer, emb_dim, num_tasks, JK, graph_pooling, molecule_node_model=None):
        super(GNN_graphpred, self).__init__()

        if num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_node_model = molecule_node_model
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.JK = JK

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        print("Loading from {} ...".format(model_file))
        state_dict = torch.load(model_file)
        self.molecule_node_model.load_state_dict(state_dict)
        return

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_node_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)
        return graph_representation, output

class MoleculeSTM(MolEncoder, TextEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config["structure"]["name"] == "magamolbart":
            # TODO: implement megamolbart
            pass
        elif config["structure"]["name"] == "gnn":
            molecule_node_model = GNN(
                num_layer=config["structure"]["gin_num_layers"],
                emb_dim=config["structure"]["gin_hidden_dim"],
                gnn_type="gin",
                drop_ratio=config["structure"]["drop_ratio"],
                JK="last",
            )
            self.structure_encoder = GNN_graphpred(
                num_layer=config["structure"]["gin_num_layers"],
                emb_dim=config["structure"]["gin_hidden_dim"],
                graph_pooling="mean",
                JK="last",
                num_tasks=1,
                molecule_node_model=molecule_node_model
            )
        else:
            raise AttributeError
        if "ckpt" in config["structure"]:
            logger.info("Loading structure encoder from %s" % (config["structure"]["ckpt"]))
            state_dict = torch.load(config["structure"]["ckpt"], map_location="cpu")
            self.structure_encoder.load_state_dict(state_dict)

        self.text_encoder = BertModel.from_pretrained(config["text"]["bert_path"])
        if "ckpt" in config["text"]:
            logger.info("Loading text encoder from %s" % (config["text"]["ckpt"]))
            state_dict = torch.load(config["text"]["ckpt"], map_location="cpu")
            missing_keys, unexpected_keys = self.text_encoder.load_state_dict(state_dict, strict=False)
            logger.info("missing keys: " + str(missing_keys))
            logger.info("unexpected keys: " + str(unexpected_keys))

        self.structure_proj_head = nn.Linear(config["structure"]["output_dim"], config["projection_dim"])
        self.text_proj_head = nn.Linear(config["text"]["output_dim"], config["projection_dim"])
        if "structure_proj_ckpt" in config:
            logger.info("Loading structure projection head from %s" % (config["structure_proj_ckpt"]))
            state_dict = torch.load(config["structure_proj_ckpt"], map_location="cpu")
            self.structure_proj_head.load_state_dict(state_dict)
        if "text_proj_ckpt" in config:
            logger.info("Loading text projection head from %s" % (config["text_proj_ckpt"]))
            state_dict = torch.load(config["text_proj_ckpt"], map_location="cpu")
            self.text_proj_head.load_state_dict(state_dict)
        self.norm = False

    def encode_mol(self, structure, proj=False, return_node_feats=False):
        mol_embeds, node_embeds = self.structure_encoder(structure)
        if proj:
            mol_embeds = self.structure_proj_head(mol_embeds)
        if not return_node_feats:
            return mol_embeds
        else:
            return mol_embeds, node_embeds

    def encode_text(self, text, proj=False):
        text_embeds = self.text_encoder(text["input_ids"], attention_mask=text["attention_mask"])["pooler_output"]
        if proj:
            return self.text_proj_head(text_embeds)
        else:
            return text_embeds