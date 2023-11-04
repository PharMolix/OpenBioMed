import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput

from models.multimodal.molt5 import MolT5
from models import SUPPORTED_MOL_ENCODER

from utils.mol_utils import convert_pyg_batch
from models.multimodal.molkformer.mol_kformer import MolKFormer


class MoleditModel(nn.Module):
    def __init__(self, config):
        super(MoleditModel, self).__init__()
        self.model = SUPPORTED_MOL_ENCODER[config["graph"]["name"]](config["graph"])
        self.ckpt = torch.load(config["graph"]["init_checkpoint"], map_location="cpu")
        if config["graph"]["name"] == "molkformer":
          self.ckpt = self.ckpt["model"]
        self.model.load_state_dict(self.ckpt, strict=False)
        self.use_molkformer = True if config["graph"]["name"] == "molkformer" else False
        self.use_momu = True if config["graph"]["name"] == "momu" else False

    def forward(self, mol):
        h, encoder_attention_mask = self.encode(mol)
        return h, encoder_attention_mask


    def decode(self, mol, num_beams, max_length):
        h, encoder_attention_mask = self.encode(mol)
        return self.generate_model.decode(
            encoder_outputs=h,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

    def encode(self, mol):
        if "input_ids" in mol:
            h = self.model.encode_text(mol, proj=True)
            encoder_attention_mask = 1
        else:
            if self.use_molkformer==True:
                # mol={"structure":{"Graph":mol}}
                mol={"structure":mol}
                graph_feats = self.model.encode_mol(mol, proj=True)
                h = graph_feats.mean(dim=1)
            if self.use_momu==True:
                graph_feats = self.model.encode_mol(mol, proj=True)
                h = graph_feats
            encoder_attention_mask = 1
        return h, encoder_attention_mask