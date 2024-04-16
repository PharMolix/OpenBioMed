import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import json

from transformers import AutoModel
from open_biomed.models import SUPPORTED_MOL_ENCODER, SUPPORTED_PROTEIN_ENCODER
from open_biomed.models.predictor import MLP

class DTIModel(nn.Module):
    def __init__(self, config, pred_dim):
        super(DTIModel, self).__init__()
        drug_encoder_config = json.load(open(config["mol"]["config_path"], "r"))
        self.drug_encoder = SUPPORTED_MOL_ENCODER[config["mol"]["name"]](drug_encoder_config)
        if "ckpt" in drug_encoder_config:
            state_dict = torch.load(open(drug_encoder_config["ckpt"], "rb"), map_location="cpu")
            if "param_key" in drug_encoder_config:
                state_dict = state_dict[drug_encoder_config["param_key"]]
            self.drug_encoder.load_state_dict(state_dict)
            logger.info("load drug encoder from %s" % (drug_encoder_config["ckpt"]))

        protein_encoder_config = json.load(open(config["protein"]["config_path"], "r"))
        self.protein_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["name"]](protein_encoder_config)
        if "ckpt" in protein_encoder_config:
            state_dict = torch.load(open(protein_encoder_config["ckpt"], "rb"), map_location="cpu")
            if "param_key" in protein_encoder_config:
                state_dict = state_dict[protein_encoder_config["param_key"]]
            self.protein_encoder.load_state_dict(state_dict)
            logger.info("load protein encoder from %s" % (protein_encoder_config["ckpt"]))

        self.pred_head = MLP(config["pred_head"], self.drug_encoder.output_dim + self.protein_encoder.output_dim, pred_dim)
    
    def forward(self, drug, protein):
        h_drug = self.drug_encoder.encode_mol(drug)
        h_protein = self.protein_encoder.encode_protein(protein)
        h = torch.cat((h_drug, h_protein), dim=1)
        return self.pred_head(h)

class DeepEIK4DTI(nn.Module):
    def __init__(self, config, pred_dim):
        super(DeepEIK4DTI, self).__init__()
        self.use_attention = config["use_attention"]
        self.projection_dim = config["projection_dim"]

        drug_encoder_config = json.load(open(config["mol"]["structure"]["config_path"], "r"))
        self.drug_structure_encoder = SUPPORTED_MOL_ENCODER[config["mol"]["structure"]["name"]](drug_encoder_config)
        protein_encoder_config = json.load(open(config["protein"]["structure"]["config_path"], "r"))
        self.protein_structure_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["structure"]["name"]](protein_encoder_config)
        self.structure_hidden_dim = self.drug_structure_encoder.output_dim + self.protein_structure_encoder.output_dim

        self.kg_project = nn.Sequential(
            nn.Linear(config["mol"]["kg"]["embedding_dim"] + config["protein"]["kg"]["embedding_dim"], self.projection_dim),
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
        h_drug_structure = self.drug_structure_encoder.encode_mol(drug["structure"])
        h_protein_structure = self.protein_structure_encoder.encode_protein(protein["structure"])
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
        