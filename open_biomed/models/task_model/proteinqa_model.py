import json

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models import SUPPORTED_TEXT_DECODER
from open_biomed.models.multimodal import BioMedGPTV, BioT5

class ProteinQAJointRepModel(nn.Module):
    SUPPORTED_JOINT_REP_MODEL = {
        "biomedgpt": BioMedGPTV,
        "biot5": BioT5,
    }

    def __init__(self, config):
        super(ProteinQAJointRepModel, self).__init__()
        self.config = config
        encoder_cls = self.SUPPORTED_JOINT_REP_MODEL[config["encoder"]["name"]]
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

        if config["encoder"]["name"] != config["text_decoder"]["name"]:
            self.decoder = SUPPORTED_TEXT_DECODER[config["text_decoder"]["name"]](config["text_decoder"])
            self.encoder_decoder_proj = nn.Linear(self.encoder.output_dim, self.decoder.hidden_size)
        else:
            self.decoder = self.encoder
            self.encoder_decoder_proj = None
        self.decoder_tokenizer = self.decoder.decoder_tokenizer

    def _concat_protein_text(self, protein, batch, text):
        bs = torch.max(batch) + 1
        num_residues = torch.sum(protein.attention_mask, dim=1)
        output_input_ids, output_attention_mask = [], []
        max_len = 0
        for b in range(bs):
            idxs = torch.where(batch == b)[0]
            output_input_ids.append(torch.cat(
                [protein.input_ids[i, :min(480 // len(idxs), num_residues[i])] for i in idxs] + [text.input_ids[b, :]], 
                dim=0
            ))
            max_len = max(max_len, output_input_ids[-1].shape[0])
        
        max_len = min(max_len, 512)
        for b in range(bs):
            if output_input_ids[b].shape[0] < max_len:
                output_attention_mask.append(torch.cat([torch.ones(output_input_ids[b].shape[0]).to(output_input_ids[b].device), torch.zeros(max_len - output_input_ids[b].shape[0]).long().to(output_input_ids[b].device)], dim=0))
                output_input_ids[b] = torch.cat((output_input_ids[b], torch.zeros(max_len - output_input_ids[b].shape[0]).long().to(output_input_ids[b].device)), dim=0)
            else:
                output_input_ids[b] = output_input_ids[b][:max_len]
                output_attention_mask.append(torch.ones(max_len).to(output_input_ids[b].device))
        return torch.stack(output_input_ids, dim=0), torch.stack(output_attention_mask, dim=0)

    def _encode(self, protein, batch, question):
        input_ids, input_attention_mask = self._concat_protein_text(protein, batch, question)
        encoder_outputs = self.encoder.encode_text({
            "input_ids": input_ids, 
            "attention_mask": input_attention_mask
        })
        attention_mask = input_attention_mask

        if self.encoder_decoder_proj is not None:
            encoder_outputs = self.encoder_decoder_proj(encoder_outputs)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
            attentions=None
        )
        return encoder_outputs, attention_mask
        
    def forward(self, protein, batch, question, answer):
        encoder_outputs, attention_mask = self._encode(protein, batch, question)
        labels = answer.input_ids.masked_fill(~answer.attention_mask.bool(), -100)
        # print(encoder_outputs.last_hidden_state.shape, labels.shape)
        return self.decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=answer.attention_mask,
            labels=labels
        )

    def generate(self, protein, batch, question, num_beams=5, max_length=256):
        encoder_outputs, attention_mask = self._encode(protein, batch, question)
        return self.decoder.decode(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

SUPPORTED_PROTEINQA_MODELS = {
    "joint_rep": ProteinQAJointRepModel
}