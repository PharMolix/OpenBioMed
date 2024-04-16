import torch
import torch.nn as nn

from open_biomed.models import SUPPORTED_CELL_ENCODER

class CTCModel(nn.Module):
    def __init__(self, config, num_labels):
        super(CTCModel, self).__init__()
        self.encoder_name = config["structure"]["name"]
        self.encoder = SUPPORTED_CELL_ENCODER[config["structure"]["name"]](**config["structure"])
        ckpt = torch.load(config["structure"]["ckpt_path"])
        if config["structure"]["param_key"] != "":
            ckpt = ckpt[config["structure"]["param_key"]]
        self.encoder.load_state_dict(ckpt)
        self.conv = nn.Conv2d(1, 1, (1, config["structure"]["dim"]))
        self.act = nn.ReLU()
        self.pred_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config["structure"]["gene_num"] + 1, config["pred_head"]["hidden_size"][0])
        )
        config["pred_head"]["hidden_size"] += [num_labels]
        for i in range(len(config["pred_head"]["hidden_size"]) - 1):
            self.pred_head.append(nn.ReLU())
            self.pred_head.append(nn.Dropout(config["pred_head"]["dropout"]))
            self.pred_head.append(nn.Linear(config["pred_head"]["hidden_size"][i], config["pred_head"]["hidden_size"][i + 1]))

    def forward(self, data):
        batch_size = data.shape[0]
        if self.encoder_name == "celllm":
            h, gene_pos = self.encoder(data, return_encodings=True)
        else:
            h = self.encoder(data, return_encodings=True)
        h = h[:,None,:,:]
        h = self.conv(h)
        h = self.act(h)
        h = h.view(h.shape[0], -1)

        if self.encoder_name == "celllm":
            pad_gene_id = self.encoder.pad_gene_id
            gene_num = pad_gene_id - 1
            out_emb = torch.zeros(batch_size, gene_num + 1).to(h.device) # , dim
            for batch in range(batch_size):
                seq_len = (gene_pos[batch] != pad_gene_id).sum()
                out_emb[batch][gene_pos[batch][:seq_len]] = h[batch][:seq_len]
            h = out_emb
        return self.pred_head(h)