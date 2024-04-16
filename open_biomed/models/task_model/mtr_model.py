import json

import torch
import torch.nn as nn

from open_biomed.models import SUPPORTED_MOL_ENCODER, SUPPORTED_TEXT_ENCODER

class MTRModel(nn.Module):
    def __init__(self, config):
        super(MTRModel, self).__init__()
        mol_config = json.load(open(config["structure"]["config_path"], "r"))
        text_config = json.load(open(config["text"]["config_path"], "r"))
        self.mol_encoder = SUPPORTED_MOL_ENCODER[config["structure"]["name"]](mol_config)
        self.text_encoder = SUPPORTED_TEXT_ENCODER[config["text"]["name"]](text_config)
        self.mol_proj = nn.Linear(self.mol_encoder.output_dim, config["projection_dim"])
        self.text_proj = nn.Linear(self.text_encoder.output_dim, config["projection_dim"])

    def encode_mol(self, mol):
        h = self.mol_encoder.encode_mol(mol)
        return self.mol_proj(h)

    def encode_text(self, text):
        h = self.text_encoder.encode_text(text)
        return self.text_proj(h)