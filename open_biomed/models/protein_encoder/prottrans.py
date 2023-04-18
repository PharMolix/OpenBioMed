import torch
import torch.nn as nn

from transformers import AutoModel

class ProtTrans(nn.Module):
    def __init__(self, config):
        super(ProtTrans, self).__init__()
        self.max_length = config["max_length"]
        self.output_dim = config["output_dim"]
        self.model = AutoModel.from_pretrained(config["model_name_or_path"])
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, self.output_dim)

    def forward(self, prot):
        batch_size, model_max_length = prot["input_ids"].shape
        h = self.model(**prot).last_hidden_state
        h = h[:, 0, :].view(batch_size, self.max_length // model_max_length, -1)
        h = torch.mean(h, dim=1)
        h = self.dropout(h)
        return self.fc(h)