import torch
from transformers import BertModel

from open_biomed.models.base_models import CellEncoder

class GeneFormer(CellEncoder):
    def __init__(self, config):
        super(GeneFormer, self).__init__()
        self.main_model = BertModel.from_pretrained(config["model_name_or_path"])
        self.output_dim = self.main_model.config.hidden_size

    def forward(self, cell):
        return self.main_model(**cell).last_hidden_state

    def encode_cell(self, cell):
        return self.forward(cell)