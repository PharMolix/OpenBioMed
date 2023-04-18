import torch
import torch.nn as nn

from models.cell_encoder import SUPPORTED_CELL_ENCODER

class CTCModel(nn.Module):
    def __init__(self, config, num_labels):
        super(CTCModel, self).__init__()
        self.encoder = SUPPORTED_CELL_ENCODER[config["structure"]["name"]](**config["structure"])
        ckpt = torch.load(config["structure"]["ckpt_path"])
        if config["structure"]["param_key"] != "":
            ckpt = ckpt[config["structure"]["param_key"]]
        self.encoder.load_state_dict(ckpt)
        self.conv = nn.Conv2d(1, 1, (1, config["structure"]["dim"]))
        self.pred_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config["structure"]["max_seq_len"], config["pred_head"]["hidden_size"][0])
        )
        config["pred_head"]["hidden_size"] += [num_labels]
        for i in range(len(config["pred_head"]["hidden_size"]) - 1):
            self.pred_head.append(nn.ReLU())
            self.pred_head.append(nn.Dropout(config["pred_head"]["dropout"]))
            self.pred_head.append(nn.Linear(config["pred_head"]["hidden_size"][i], config["pred_head"]["hidden_size"][i + 1]))

    def forward(self, data):
        h = self.encoder(data, return_encodings=True)
        h = h[:,None,:,:]
        h = self.conv(h)
        h = h.view(h.shape[0], -1)
        return self.pred_head(h)