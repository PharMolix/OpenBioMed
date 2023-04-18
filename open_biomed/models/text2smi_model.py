import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput

from models.drug_encoder import MoMu, KVPLM
from models.text_encoder import BaseTransformers
from models.drug_decoder.molt5 import MolT5

SUPPORTED_ENCODER = {
    "SciBERT": BaseTransformers,
    "MoMu": MoMu,
    "KVPLM": KVPLM
}

class Text2SMILESModel(nn.Module):
    def __init__(self, config):
        super(Text2SMILESModel, self).__init__()
        self.generate_model = MolT5(config["smiles"])
        if "text" in config:
            self.text_encoder = SUPPORTED_ENCODER[config["text"]["name"]](config["text"])
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

    def forward(self, mol):
        h, encoder_attention_mask = self._encode_text(mol)
        labels = mol["structure"]["input_ids"].masked_fill(~mol["structure"]["attention_mask"].bool(), -100)
        return self.generate_model(
            encoder_outputs=h,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=mol["structure"]["attention_mask"],
            labels=labels
        )

    def decode(self, mol, num_beams, max_length):
        h, encoder_attention_mask = self._encode_text(mol)
        return self.generate_model.decode(
            encoder_outputs=h,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

    def _encode_text(self, mol):
        if self.text_encoder is not None:
            text_feats = self.text_encoder.encode_text(mol["text"], return_cls=False, proj=False)
            text_feats = self.text_projector(text_feats)
        else:
            text_feats = self.generate_model.encode(mol["text"])
        h = BaseModelOutput(
            last_hidden_state=text_feats,
            hidden_states=None,
            attentions=None
        )
        text_attention_mask = mol["text"]["attention_mask"]
        return h, text_attention_mask