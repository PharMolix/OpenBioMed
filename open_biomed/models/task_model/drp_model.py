import torch
import torch.nn as nn
from models.molecule.gin_tgsa import GINTGSA
from models.cell import CellGAT
from models.cell import SUPPORTED_CELL_ENCODER

class ConvPooler(nn.Module):
    def __init__(self, dim, full_seq_len):
        super().__init__()
        self.full_seq_len = full_seq_len
        self.pad_gene_id = full_seq_len
        self.conv = nn.Conv2d(1, 1, (1, dim))
    def forward(self, h, gene_pos):
        batchsize, seqlen, dim = h.shape
        h = h[:,None,:,:]
        h = self.conv(h)
        h = h.view(h.shape[0], -1)

        out_emb = torch.zeros(batchsize, self.full_seq_len).to(h.device)#, dim
        for batch in range(batchsize):
            seq_len = (gene_pos[batch] != self.pad_gene_id).sum()
            out_emb[batch][gene_pos[batch][:seq_len]] = h[batch][:seq_len]
        h = out_emb
        return h
    
class ConvPoolerShort(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, (1, dim))
    def forward(self, h):
        h = h[:,None,:,:]
        h = self.conv(h)
        h = h.view(h.shape[0], -1)
        return h
    
class TGDRP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_drug = config["layer_drug"]
        self.dim_drug = config["dim_drug"]
        self.input_dim_cell = config["input_dim_cell"]
        self.layer_cell = config["layer_cell"]
        self.dim_cell = config["dim_cell"]
        self.dropout = config["dropout"]
        self.cell_encoder_config = config["cell_encoder"]

    def _build(self):
        # drug graph branch
        self.GNN_drug = GINTGSA(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        # cell graph branch
        if self.cell_encoder_config["name"] == 'gat':
            self.cell_encoder = CellGAT(self.input_dim_cell, self.layer_cell, self.dim_cell, self.cluster_predefine)
            cell_encode_dim = self.dim_cell * self.cell_encoder.final_node
        elif self.cell_encoder_config["name"] == 'deepcdr':
            self.cell_encoder = SUPPORTED_CELL_ENCODER[self.cell_encoder_config["name"]](**self.cell_encoder_config)
            cell_encode_dim = self.cell_encoder_config['output_dim']
        else:
            self.cell_encoder = SUPPORTED_CELL_ENCODER[self.cell_encoder_config["name"]](**self.cell_encoder_config)
            if "ckpt_path" in self.cell_encoder_config:
                ckpt = torch.load(self.cell_encoder_config["ckpt_path"])
                if self.cell_encoder_config["param_key"] != "":
                    ckpt = ckpt[self.cell_encoder_config["param_key"]]
                self.cell_encoder.load_state_dict(ckpt)
            if self.cell_encoder_config["name"] in ["scbert", "celllm"]:
                if self.cell_encoder_config["name"] == "celllm":
                    self.cell_encoder.to_out = ConvPooler(self.cell_encoder_config["dim"], self.cell_encoder_config["gene_num"] + 1)
                    cell_encode_dim = self.cell_encoder_config["gene_num"] + 1 
                else:
                    self.cell_encoder.to_out = ConvPoolerShort(self.cell_encoder_config["dim"])
                    cell_encode_dim = self.cell_encoder_config["max_seq_len"]

            else:
                cell_encode_dim = self.dim_cell * self.cell_encoder.final_node

        self.cell_emb = nn.Sequential(
            nn.Linear(cell_encode_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x_drug = self.GNN_drug(drug)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        x_cell = self.cell_encoder(cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x


