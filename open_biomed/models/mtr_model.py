import json

import torch
import torch.nn as nn

from models.drug_encoder import SUPPORTED_DRUG_ENCODER
from models.text_encoder import SUPPORTED_TEXT_ENCODER

class MTRModel(nn.Module):
    def __init__(self, config):
        super(MTRModel, self).__init__()
        drug_config = json.load(open(config["structure"]["config_path"], "r"))
        text_config = json.load(open(config["text"]["config_path"], "r"))
        self.drug_encoder = SUPPORTED_DRUG_ENCODER[config["structure"]["name"]](**drug_config)
        self.text_encoder = SUPPORTED_TEXT_ENCODER[config["text"]["name"]](text_config)
        self.drug_proj = nn.Linear(self.drug_encoder.output_dim, config["projection_dim"])
        self.text_proj = nn.Linear(self.text_encoder.output_dim, config["projection_dim"])

    def encode_structure(self, drug):
        h, _ = self.drug_encoder(drug)
        return self.drug_proj(h)

    def encode_text(self, text):
        h = self.text_encoder(text)
        return self.text_proj(h)