import json
import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models import SUPPORTED_TEXT_ENCODER, SUPPORTED_TEXT_DECODER

class Text2SMILESModel(nn.Module):
    def __init__(self, config):
        super(Text2SMILESModel, self).__init__()
        self.generate_model = SUPPORTED_TEXT_DECODER[config["structure"]["name"]](config["structure"])
        if "text" in config:
            if "config_path" in config["text"]:
                text_config = json.load(open(config["text"]["config_path"]))
                text_config["name"] = config["text"]["name"]
                text_config["use_num_layers"] = config["text"]["use_num_layers"]
            else:
                text_config = config["text"]
            self.text_encoder = SUPPORTED_TEXT_ENCODER[config["text"]["name"]](text_config)
            if "init_checkpoint" in config["text"]:
                ckpt = torch.load(config["text"]["init_checkpoint"])
                if "param_key" in config["text"]:
                    ckpt = ckpt[config["text"]["param_key"]]
                self.text_encoder.load_state_dict(ckpt)
            self.text_projector = nn.Sequential(
                nn.Linear(config["text"]["output_dim"], self.generate_model.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.generate_model.hidden_size, self.generate_model.hidden_size)
            )
        else:
            self.text_encoder = None
        self.decoder_tokenizer = self.generate_model.decoder_tokenizer

    def forward(self, text, mol):
        h, attention_mask = self._encode_text(text)
        labels = mol["input_ids"].masked_fill(~mol["attention_mask"].bool(), -100)
        return self.generate_model(
            encoder_outputs=h,
            attention_mask=attention_mask,
            decoder_attention_mask=mol["attention_mask"],
            labels=labels
        )

    def decode_mol(self, text, num_beams, max_length):
        h, attention_mask = self._encode_text(text)
        return self.generate_model.decode(
            encoder_outputs=h,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

    def _encode_text(self, text):
        if self.text_encoder is not None:
            text_feats = self.text_encoder.encode_text(text, return_cls=False, proj=False)
            text_feats = self.text_projector(text_feats)
        else:
            text_feats = self.generate_model.encode_text(text)
        h = BaseModelOutput(
            last_hidden_state=text_feats,
            hidden_states=None,
            attentions=None
        )
        text_attention_mask = text["attention_mask"]
        return h, text_attention_mask

    def mol_generation_loss(self, text, mol):
        return self.forward(text, mol)