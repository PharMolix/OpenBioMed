import logging
logger = logging.getLogger(__name__)

import json
import torch
import torch.nn as nn

from torch_geometric.data import Batch
from transformers.modeling_outputs import BaseModelOutput

from models import SUPPORTED_MOL_ENCODER, SUPPORTED_TEXT_ENCODER, SUPPORTED_TEXT_DECODER
from models.multimodal import KVPLM, MolT5, MolFM, DrugFM
from utils.mol_utils import convert_pyg_batch

class MolQASepRepModel(nn.Module):
    def __init__(self, config):
        super(MolQASepRepModel, self).__init__()
        self.config = config
        if "config_path" in config["mol"]:
            mol_encoder_config = json.load(open(config["mol"]["config_path"], "r"))
        else:
            mol_encoder_config = config["mol"]
        #for key, value in mol_encoder_config.items():
        #    self.config["mol"][key] = value
        self.mol_encoder = SUPPORTED_MOL_ENCODER[config["mol"]["name"]](mol_encoder_config)
        if "init_checkpoint" in config["mol"]:
            logger.info("Loading molecule checkpoint from %s" % (config["mol"]["init_checkpoint"]))
            state_dict = torch.load(config["mol"]["init_checkpoint"], map_location="cpu")
            if "param_key" in config["mol"]:
                state_dict = state_dict[config["mol"]["param_key"]]
            self.mol_encoder.load_state_dict(state_dict)

        if config["text_encoder"]["name"] == config["mol"]["name"]:
            self.text_encoder = self.mol_encoder
        else:
            self.text_encoder = SUPPORTED_TEXT_ENCODER[config["text_encoder"]["name"]](config["text_encoder"])
        self.mol_proj = nn.Linear(self.mol_encoder.output_dim, self.text_encoder.output_dim)
        if config["text_decoder"]["name"] == config["text_encoder"]["name"]:
            self.text_decoder = self.text_encoder
            self.encoder_decoder_proj = None
        else:
            self.text_decoder = SUPPORTED_TEXT_DECODER[config["text_decoder"]["name"]](config["text_decoder"])
            self.encoder_decoder_proj = nn.Linear(self.text_encoder.output_dim, self.text_decoder.hidden_size)

    def _concat_mol_text(self, mol_embeds, mol_attention_mask, text_embeds, text_attention_mask):
        # put <cls> first
        bs = mol_embeds.shape[0]
        num_atoms = torch.sum(mol_attention_mask, dim=1).int()
        output_embeds, output_attention_mask = [], []
        for i in range(bs):
            output_embeds.append(torch.cat((
                text_embeds[i, 0, :].unsqueeze(0),
                mol_embeds[i, :num_atoms[i], :],
                text_embeds[i, 1:, :],
                mol_embeds[i, num_atoms[i]:, :]
            ), dim=0))
            output_attention_mask.append(torch.cat((
                text_attention_mask[i, 0].unsqueeze(0),
                mol_attention_mask[i, :num_atoms[i]],
                text_attention_mask[i, 1:],
                mol_attention_mask[i, num_atoms[i]:]
            ), dim=0))
        return torch.stack(output_embeds, dim=0), torch.stack(output_attention_mask, dim=0)

    def _encode(self, mol, question):
        _, mol_outputs = self.mol_encoder.encode_mol(mol, return_node_feats=True)
        #mol_outputs = self.mol_encoder.encode_mol(mol)
        if isinstance(mol, Batch):
            mol_embeds, mol_attention_mask = convert_pyg_batch(mol_outputs, mol.batch, self.config["mol"]["max_n_nodes"])
        else:
            mol_embeds = mol_outputs
            mol_attention_mask = mol.attention_mask
        mol_embeds = self.mol_proj(mol_embeds)
        text_embeds = self.text_encoder.main_model.get_input_embeddings()(question.input_ids)
        embeds, attention_mask = self._concat_mol_text(mol_embeds, mol_attention_mask, text_embeds, question.attention_mask)
        #text_outputs = self.text_encoder.encode_text(question)
        #encoder_outputs, encoder_attention_mask = self._concat_mol_text(mol_embeds, mol_attention_mask, text_outputs.last_hidden_state, question.attention_mask)
        encoder_outputs = self.text_encoder.main_model.encoder(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
            attentions=None
        )
        return encoder_outputs, attention_mask

    def forward(self, mol, question, answer):
        encoder_outputs, encoder_attention_mask = self._encode(mol, question)
        if self.encoder_decoder_proj is not None:
            encoder_outputs = self.encoder_decoder_proj(encoder_outputs)
        labels = answer.input_ids.masked_fill(~answer.attention_mask.bool(), -100)
        return self.text_decoder(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=answer.attention_mask,
            labels=labels
        )

    def generate(self, mol, question, num_beams=5, max_length=256):
        encoder_outputs, encoder_attention_mask = self._encode(mol, question)
        return self.text_decoder.decode(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )

class MolQAJointRepModel(nn.Module):
    SUPPORTED_JOINT_REP_MODEL = {
        "kvplm": (KVPLM, "early"),
        "kvplm*": (KVPLM, "early"),
        "molt5": (MolT5, "early"),
        "molfm": (MolFM, "feature"),
        "drugfm": (DrugFM, "feature")
    }

    def __init__(self, config):
        super(MolQAJointRepModel, self).__init__()
        self.config = config
        encoder_cls, fusion_type = self.SUPPORTED_JOINT_REP_MODEL[config["encoder"]["name"]]
        if "config_path" in config["encoder"]:
            encoder_config = json.load(open(config["encoder"]["config_path"], "r"))
        else:
            encoder_config = config["encoder"]
        self.encoder = encoder_cls(encoder_config)
        if "init_checkpoint" in config["encoder"]:
            state_dict = torch.load(config["encoder"]["init_checkpoint"], map_location="cpu")
            if "param_key" in config["encoder"]:
                state_dict = state_dict[config["encoder"]["param_key"]]
            self.encoder.load_state_dict(state_dict)
        self.fusion_type = fusion_type
        if config["encoder"]["name"] != config["text_decoder"]["name"]:
            self.decoder = SUPPORTED_TEXT_DECODER[config["text_decoder"]["name"]](config["text_decoder"])
            self.encoder_decoder_proj = nn.Linear(self.encoder.output_dim, self.decoder.hidden_size)
        else:
            self.decoder = self.encoder
            self.encoder_decoder_proj = None

    def _concat_smi_text(self, mol, text):
        bs = mol.input_ids.shape[0]
        num_atoms = torch.sum(mol.attention_mask, dim=1)
        output_input_ids, output_attention_mask = [], []
        # ignore <eos> in molecule and <cls> in text
        for i in range(bs):
            output_input_ids.append(torch.cat((
                mol.input_ids[i, :num_atoms[i] - 1],
                text.input_ids[i, 1:],
                mol.input_ids[i, num_atoms[i]:]
            ), dim=0))
            output_attention_mask.append(torch.cat((
                mol.attention_mask[i, :num_atoms[i] - 1],
                text.attention_mask[i, 1:],
                mol.attention_mask[i, num_atoms[i]:]
            ), dim=0))
        return torch.stack(output_input_ids, dim=0), torch.stack(output_attention_mask, dim=0)

    def _encode(self, mol, question):
        if self.fusion_type == "early":
            input_ids, input_attention_mask = self._concat_smi_text(mol, question)
            encoder_outputs = self.encoder.encode({
                "input_ids": input_ids, 
                "attention_mask": input_attention_mask
            })
            encoder_attention_mask = input_attention_mask
        elif self.fusion_type == "feature":
            encoder_outputs = self.encoder(mol, question).last_hidden_state
            encoder_attention_mask = question.attention_mask
        if self.encoder_decoder_proj is not None:
            encoder_outputs = self.encoder_decoder_proj(encoder_outputs)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
            attentions=None
        )
        return encoder_outputs, encoder_attention_mask
        
    def forward(self, mol, question, answer):
        encoder_outputs, encoder_attention_mask = self._encode(mol, question)
        labels = answer.input_ids.masked_fill(~answer.attention_mask.bool(), -100)
        return self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=answer.attention_mask,
            labels=labels
        )

    def generate(self, mol, question, num_beams=5, max_length=256):
        encoder_outputs, encoder_attention_mask = self._encode(mol, question)
        return self.decoder.decode(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

SUPPORTED_MOLQA_MODELS = {
    "sep_rep": MolQASepRepModel,
    "joint_rep": MolQAJointRepModel,
}