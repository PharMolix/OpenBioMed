import torch
import torch.nn as nn
import json

from transformers import AutoModel
from models.drug_encoder.momu import MoMu
from models.drug_encoder.cnn import CNN
from models.drug_encoder.molclr_gnn import GINet
from models.drug_encoder.pyg_gnn import PygGNN
from models.drug_encoder.biomedgpt import BioMedGPT
from models.drug_encoder.kv_plm import KVPLM


SUPPORTED_DRUG_ENCODER = {
    "cnn": CNN,
    "graphcl": MoMu,
    "molclr": GINet,
    "graphmvp": PygGNN,
    "biomedgpt": BioMedGPT,
    "kvplm": KVPLM
}


activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
                    self.model.append(nn.BatchNorm1d(hidden_dims[i + 1]))
    
    def forward(self, h):
        return self.model(h)
      

# TODO: choose header for different encoder
HEAD4ENCODER = {
    "deepeik": MLP,
    "momu": nn.Linear,
    "molclr": nn.Linear,
    "graphmvp": nn.Linear,
    "biomedgpt": nn.Linear,
    "kvplm": MLP
}


class DPModel(nn.Module):

    def __init__(self, config, out_dim):
        super(DPModel, self).__init__()
        # prepare model
        self.name = config["model"]
        self.model_num = len(config["data"]["drug"]["modality"]) 
        self.config = config
        if config["model"] in ["DeepEIK", "biomedgpt"]:
            self.encoder = SUPPORTED_DRUG_ENCODER[config["model"]](config["network"])
        elif config["model"] in ["graphcl",  "molalbef", "kvplm"]:
            self.encoder = SUPPORTED_DRUG_ENCODER[config["model"]](config["network"]["structure"])
            # TODO: hard code for signle module of molalbef which used func encode_structure
            if len(config["data"]["drug"]["modality"]) == 1:
                self.encoder.output_dim = self.encoder.gin_hidden_dim
        else:
            self.encoder = SUPPORTED_DRUG_ENCODER[config["model"]](**config["network"]["structure"])
        try:
            encoder_ckpt = config["network"]["structure"]["init_checkpoint"]
        except:
            encoder_ckpt = ""
        if encoder_ckpt != "":
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            param_key = config["network"]["structure"]["param_key"]
            if param_key != "":
                ckpt = ckpt[param_key]
            # self.encoder.load_state_dict(ckpt)
            missing_keys, unexpected_keys = self.encoder.load_state_dict(ckpt, strict=False)
            print("encoder missing_keys: ", missing_keys)
            print("encoder unexpected_keys: ", unexpected_keys)
            
        self.proj_head = HEAD4ENCODER[config["network"]["structure"]["name"]]
        if issubclass(self.proj_head, nn.Linear):
            self.proj_head = self.proj_head(self.encoder.output_dim, out_dim)
        else:
            self.proj_head = self.proj_head(config["network"]["pred_head"], self.encoder.output_dim, out_dim)
        
        self.params_count = count_parameters(self)
        print("params_count: ", self.params_count)
        
          
    def forward(self, drug):
        if hasattr(self.encoder, "encode_structure") and not isinstance(self.encoder, BioMedGPT):
            h = self.encoder.encode_structure(drug)  # Momu encoder_struct
        elif isinstance(self.encoder, BioMedGPT):
            h = self.encoder.encode_structure_with_text(drug["structure"], drug["text"], proj=True)  # encoder_struct_with_text
        elif len(self.config["data"]["drug"]["modality"]) == 1 and self.name == "molalbef":
            h = self.encoder.encode_structure(drug, proj=False)
        elif len(self.config["data"]["drug"]["modality"]) == 3 and self.name == "molalbef":
            # h = self.encoder.encode_structure_with_kg(drug["structure"], drug["kg"])  # encoder_struct_with_kg
            # h = self.encoder.encode_all_module(drug["structure"], drug["kg"], drug["text"])
            h = self.encoder.encode_structure_with_all(drug["structure"], drug["text"], drug["kg"])
            # h, _ = self.encoder.forward(drug["structure"], drug["text"], drug["kg"])
        else:
            h, _ = self.encoder(drug)  # encoder_struct
        return self.proj_head(h)
    

class DeepEIK4DP(nn.Module):
    def __init__(self, config, out_dim):
        super(DeepEIK4DP, self).__init__()
        self.name = "deepeik"
        self.model_num = 3
        self.use_attention = config["use_attention"]
        self.projection_dim = config["projection_dim"]
        
        drug_encoder_config = json.load(open(config["structure"]["config_path"], "r"))
        self.drug_structure_encoder = SUPPORTED_DRUG_ENCODER[config["structure"]["name"]](**drug_encoder_config)
        if "init_checkpoint" in drug_encoder_config.keys():
            encoder_ckpt = drug_encoder_config["init_checkpoint"]
            assert encoder_ckpt
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            param_key = drug_encoder_config["param_key"]
            if param_key != "":
                ckpt = ckpt[param_key]
            # self.encoder.load_state_dict(ckpt)
            missing_keys, unexpected_keys = self.drug_structure_encoder.load_state_dict(ckpt, strict=False)
            print("encoder missing_keys: ", missing_keys)
            print("encoder unexpected_keys: ", unexpected_keys)
            
        self.structure_hidden_dim = self.drug_structure_encoder.output_dim
        
        self.kg_project = nn.Sequential(
                                        nn.Linear(config["kg"]["embedding_dim"], self.projection_dim),
                                        nn.Dropout(config["projection_dropout"])
                                        )

        # TODO: need to update based on different text_tokenizer
        if "text" in config:
            self.text_encoder = AutoModel.from_pretrained(config["text"]["model_name_or_path"])
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
        
        self.params_count = count_parameters(self)
        print("params_count: ", self.params_count)
        
    def forward(self, drug):
        # TODO: 这里是兼容molclr的，molclr的输出是个tuple，现在使用的第一个输出
        self.h_drug_structure, _ = self.drug_structure_encoder(drug["structure"])
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
