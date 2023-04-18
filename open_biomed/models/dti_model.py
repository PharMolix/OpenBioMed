import torch
import torch.nn as nn
import json

from transformers import AutoModel
from models.drug_encoder import SUPPORTED_DRUG_ENCODER
from models.protein_encoder import SUPPORTED_PROTEIN_ENCODER

activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}

class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        hidden_dims = [input_dim] + config["hidden_size"] + [output_dim]
        for i in range(len(hidden_dims) - 1):
            self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                self.model.append(nn.Dropout(config["dropout"]))
                if config["activation"] != "none":
                    self.model.append(activation[config["activation"]])
                if config["batch_norm"]:
                    self.model.append(nn.BatchNorm1d())
    
    def forward(self, h):
        return self.model(h).squeeze()


class DTIModel(nn.Module):
    def __init__(self, config, pred_dim):
        super(DTIModel, self).__init__()
        drug_encoder_config = json.load(open(config["drug"]["config_path"], "r"))
        self.drug_encoder = SUPPORTED_DRUG_ENCODER[config["drug"]["name"]](drug_encoder_config)

        protein_encoder_config = json.load(open(config["protein"]["config_path"], "r"))
        self.protein_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["name"]](protein_encoder_config)

        self.pred_head = MLP(config["pred_head"], self.drug_encoder.output_dim + self.protein_encoder.output_dim, pred_dim)
    
    def forward(self, drug, protein):
        h_drug = self.drug_encoder(drug)
        h_protein = self.protein_encoder(protein)
        h = torch.cat((h_drug, h_protein), dim=1)
        return self.pred_head(h)

class DeepEIK4DTI(nn.Module):
    def __init__(self, config, pred_dim):
        super(DeepEIK4DTI, self).__init__()
        self.use_attention = config["use_attention"]
        self.projection_dim = config["projection_dim"]

        drug_encoder_config = json.load(open(config["drug"]["structure"]["config_path"], "r"))
        self.drug_structure_encoder = SUPPORTED_DRUG_ENCODER[config["drug"]["structure"]["name"]](drug_encoder_config)
        protein_encoder_config = json.load(open(config["protein"]["structure"]["config_path"], "r"))
        self.protein_structure_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["structure"]["name"]](protein_encoder_config)
        self.structure_hidden_dim = self.drug_structure_encoder.output_dim + self.protein_structure_encoder.output_dim

        self.kg_project = nn.Sequential(
            nn.Linear(config["drug"]["kg"]["embedding_dim"] + config["protein"]["kg"]["embedding_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        if "text" in config:
            self.text_encoder = AutoModel.from_pretrained(config["text"]["model_name_or_path"])
        else:
            self.text_encoder = None
        self.text_project = nn.Sequential(
            nn.Linear(config["text_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        if self.use_attention:
            self.attn = nn.MultiheadAttention(self.structure_hidden_dim + self.projection_dim, num_heads=config["num_attention_heads"], kdim=self.text_encoder.hidden_dim, vdim=self.text_encoder.hidden_dim)
        self.pred_head = MLP(config["pred_head"], self.structure_hidden_dim + 2 * self.projection_dim, pred_dim)

    def forward(self, drug, protein):
        h_drug_structure = self.drug_structure_encoder(drug["structure"])
        h_protein_structure = self.protein_structure_encoder(protein["structure"])
        h_structure = torch.cat((h_drug_structure, h_protein_structure), dim=1)
        h_kg = self.kg_project(torch.cat((drug["kg"], protein["kg"]), dim=1))
        if self.text_encoder is not None:
            h_text = self.text_encoder(**drug["text"]).last_hidden_state[:, 0, :]
        else:
            h_text = drug["text"]
        if self.use_attention:
            _, attn = self.attn(torch.cat(h_structure, h_kg).unsqueeze(1), h_text, h_text)
            h_text = torch.matmul(attn * drug["text"].unsqueeze(1), h_text)
        h_text = self.text_project(h_text)
        h = torch.cat((h_structure, h_kg, h_text), dim=1)
        return self.pred_head(h)
        