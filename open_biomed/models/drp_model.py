import torch
import torch.nn as nn
from models.drug_encoder import DrugGINTGSA
from models.cell_encoder import CellGAT

class TGDRP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_drug = config["layer_drug"]
        self.dim_drug = config["dim_drug"]
        self.input_dim_cell = config["input_dim_cell"]
        self.layer_cell = config["layer_cell"]
        self.dim_cell = config["dim_cell"]
        self.dropout = config["dropout"]

    def _build(self):
        # drug graph branch
        self.GNN_drug = DrugGINTGSA(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        # cell graph branch
        self.GNN_cell = CellGAT(self.input_dim_cell, self.layer_cell, self.dim_cell, self.cluster_predefine)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 1024),
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
        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x


