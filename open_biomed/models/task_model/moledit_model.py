import torch
import torch.nn as nn

from models import SUPPORTED_MOL_ENCODER


class MoleditModel(nn.Module):
    def __init__(self, config):
        super(MoleditModel, self).__init__()
        if "smiles" in config:
            self.model = SUPPORTED_MOL_ENCODER[config["smiles"]["name"]](config["smiles"])
            self.use_molstm = True if config["smiles"]["name"] == "molstm" else False
        elif "graph" in config:
            self.model = SUPPORTED_MOL_ENCODER[config["graph"]["name"]](config["graph"])
            if config["graph"]["name"] == "molkformer":
                    self.ckpt = torch.load(config["graph"]["init_checkpoint"], map_location="cpu")
                    self.ckpt = self.ckpt["model"]
                    self.model.load_state_dict(self.ckpt, strict=False)
            if config["graph"]["name"] == "momu":
                    self.ckpt = torch.load(config["graph"]["init_checkpoint"], map_location="cpu")
                    self.model.load_state_dict(self.ckpt, strict=False)

            self.use_molkformer = True if config["graph"]["name"] == "molkformer" else False
            self.use_momu = True if config["graph"]["name"] == "momu" else False
            self.use_molstm = True if config["graph"]["name"] == "molstm" else False

    def forward(self, mol):
        h = self.encode(mol)
        return h

    def encode(self, mol):
        #text_encode
        if "input_ids" in mol:
            h = self.model.encode_text(mol, proj=True)
        #graph_encode
        else:
            if self.use_molkformer==True:
                mol={"structure":mol}
                graph_feats = self.model.encode_mol(mol, proj=True)
                h = graph_feats.mean(dim=1)
            if self.use_momu==True:
                graph_feats = self.model.encode_mol(mol, proj=True)
                h = graph_feats
            if self.use_molstm==True:
                graph_feats = self.model.encode_mol(mol, proj=True)
                h = graph_feats
        return h