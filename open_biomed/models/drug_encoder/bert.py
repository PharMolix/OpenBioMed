import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from transformers import BertModel

class DrugBERT(nn.Module):
    def __init__(self, config):
        super(DrugBERT, self).__init__()

        self.text_encoder = BertModel.from_pretrained(config["model_name_or_path"])
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, drug):
        return self.encode_structure(drug["strcture"]), self.encode_text(drug["text"])

    def encode_structure(self, structure):
        h = self.text_encoder(**structure)["pooler_output"]
        return self.dropout(h)

    def encode_text(self, text):
        h = self.text_encoder(**text)["pooler_output"]
        return self.dropout(h)