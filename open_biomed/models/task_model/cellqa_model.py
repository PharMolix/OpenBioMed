import logging
logger = logging.getLogger(__name__)

import json
import torch
import torch.nn as nn

from torch_geometric.data import Batch
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models import SUPPORTED_CELL_ENCODER, SUPPORTED_TEXT_DECODER

class CellQAModel(nn.Module):
    def __init__(self, config):
        super(CellQAModel, self).__init__()
        self.config = config
        if "config_path" in config["cell"]:
            cell_encoder_config = json.load(open(config["cell"]["config_path"], "r"))
        else:
            cell_encoder_config = config["cell"]
        self.cell_encoder = SUPPORTED_CELL_ENCODER[config["cell"]["name"]](cell_encoder_config)
        if "init_checkpoint" in config["cell"]:
            logger.info("Loading cell checkpoint from %s" % (config["cell"]["init_checkpoint"]))
            state_dict = torch.load(config["cell"]["init_checkpoint"], map_location="cpu")
            if "param_key" in config["cell"]:
                state_dict = state_dict[config["cell"]["param_key"]]
            self.cell_encoder.load_state_dict(state_dict)
        if config["cell"]["freeze"]:
            for k, v in self.cell_encoder.named_parameters():
                v.requires_grad = False

        self.num_query_tokens = config["num_query_tokens"]
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_tokens, self.cell_encoder.output_dim))
        self.pooler = nn.MultiheadAttention(embed_dim=self.cell_encoder.output_dim, num_heads=1, batch_first=True)
        self.text_model = SUPPORTED_TEXT_DECODER[config["text"]["name"]](config["text"])
        self.projector = nn.Linear(self.cell_encoder.output_dim, self.text_model.main_model.config.hidden_size)
        self.decoder_tokenizer = self.text_model.decoder_tokenizer

    def _encode(self, cell, question):
        batch_size = question.input_ids.shape[0]
        cell_outputs = self.cell_encoder.encode_cell(cell)
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        cell_embeds, _ = self.pooler(query_embeds, cell_outputs, cell_outputs, attn_mask=(1 - cell.attention_mask.unsqueeze(1).expand(-1, self.num_query_tokens, -1)).bool())
        cell_embeds = self.projector(cell_embeds)
        text_embeds = self.text_model.main_model.encoder(**question).last_hidden_state
        attention_mask = torch.cat([torch.ones(query_embeds.shape[:-1]).to(query_embeds.device), question.attention_mask], dim=1)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=torch.cat([cell_embeds, text_embeds], dim=1),
            hidden_states=None,
            attentions=None
        )
        return encoder_outputs, attention_mask

    def forward(self, cell, question, answer):
        encoder_outputs, attention_mask = self._encode(cell, question)
        labels = answer.input_ids.masked_fill(~answer.attention_mask.bool(), -100)
        return self.text_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=answer.attention_mask,
            labels=labels
        )

    def generate(self, cell, question, num_beams=5, max_length=256):
        encoder_outputs, attention_mask = self._encode(cell, question)
        return self.text_model.decode(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )

SUPPORTED_CELLQA_MODELS = {
    "composed": CellQAModel
}