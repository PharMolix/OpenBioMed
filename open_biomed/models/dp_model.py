import torch
import torch.nn as nn
import json

from transformers import AutoModel
from models.drug_encoder.momu import MoMu
from models.drug_encoder.cnn import CNN
from models.drug_encoder.molclr_gnn import GINet
from models.drug_encoder.pyg_gnn import PygGNN


SUPPORTED_DRUG_ENCODER = {
    "cnn": CNN,
    "graphcl": MoMu,
    "molclr": GINet,
    "graphmvp": PygGNN
}


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
        return self.model(h)


# TODO: choose header for different encoder
HEAD4ENCODER = {
    "deepeik": MLP,
    "momu": nn.Linear,
    "molclr": nn.Linear,
    "graphmvp": nn.Linear
}


class DPModel(nn.Module):

    def __init__(self, config, out_dim):
        super(DPModel, self).__init__()
        # prepare model
        if config["model"] == "DeepEIK":
            self.encoder = SUPPORTED_DRUG_ENCODER[config["model"]](config["network"])
        elif config["model"] == "graphcl":
            self.encoder = SUPPORTED_DRUG_ENCODER[config["model"]](config["network"]["structure"])
        else:
            self.encoder = SUPPORTED_DRUG_ENCODER[config["model"]](**config["network"]["structure"])
        encoder_ckpt = config["network"]["structure"]["init_checkpoint"]
        if encoder_ckpt != "":
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            param_key = config["network"]["structure"]["param_key"]
            if param_key != "":
                ckpt = ckpt[param_key]
                missing_keys, unexpected_keys = self.encoder.load_state_dict(ckpt, strict=False)
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
            
        self.proj_head = HEAD4ENCODER[config["network"]["structure"]["name"]](self.encoder.output_dim, out_dim)
        
    def forward(self, drug):
        if hasattr(self.encoder, "encode_structure"):
            h = self.encoder.encode_structure(drug)  # Momu encoder_struct
        else:
            h = self.encoder.encode_structure_with_kg(drug["structure"], drug["kg"])
        return self.proj_head(h)
    

class DeepEIK4DP(nn.Module):
    def __init__(self, config, out_dim):
        super(DeepEIK4DP, self).__init__()
        self.use_attention = config["use_attention"]
        self.projection_dim = config["projection_dim"]
        
        drug_encoder_config = json.load(open(config["drug"]["structure"]["config_path"], "r"))
        self.drug_structure_encoder = SUPPORTED_DRUG_ENCODER[config["drug"]["structure"]["name"]](drug_encoder_config)
        self.structure_hidden_dim = self.drug_structure_encoder.output_dim
        
        self.kg_project = nn.Sequential(
                                        nn.Linear(config["drug"]["kg"]["embedding_dim"], self.projection_dim),
                                        nn.Dropout(config["projection_dropout"])
                                        )
        
        # TODO: need to update based on different text_tokenizer
        if "text" in config["drug"]:
            self.text_encoder = AutoModel.from_pretrained(config["drug"]["text"]["model_name_or_path"])
        else:
            self.text_encoder = None
        # get the embeding of a sentence
        self.text_project = nn.Sequential(
            nn.Linear(config["text_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )
        
        if self.use_attention:
            self.attn = nn.MultiheadAttention(self.structure_hidden_dim + self.projection_dim, 
                                              num_heads=config["num_attentin_heads"],
                                              kdim=self.text_encoder.hidden_dim, 
                                              vdim=self.text_encoder.hidden_dim
                                              )
        # structure + kg + text
        self.pred_head = MLP(config["pred_head"], self.structure_hidden_dim + 2 * self.projection_dim, out_dim)
        
    def forward(self, drug):
        self.h_drug_structure = self.drug_structure_encoder(drug["structure"])
        h_kg = drug["kg"]
        
        if self.text_encoder is not None:
            h_text = self.text_encoder(**drug["text"]).last_hidden_state[:, 0, :]
        else:
            h_text = drug["text"]
        
        # TODO: 
        if self.use_attention:
            _, attn = self.attn(torch.cat(self.h_drug_structure, h_kg).unsqueeze(1), h_text, h_text)
            h_text = torch.matmul(attn * drug["text"].unsqueeze(1), h_text)
            
        h_text = self.text_project(h_text)
        h = torch.cat((self.h_drug_structure, h_kg, h_text), dim=1)
        
        return self.pred_head(h)
