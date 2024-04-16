import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5ForConditionalGeneration
from open_biomed.models.base_models import MolEncoder, TextEncoder

class MolT5(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(MolT5, self).__init__()
        self.main_model = T5ForConditionalGeneration.from_pretrained(config['model_name_or_path'])
        self.decoder_tokenizer = T5Tokenizer.from_pretrained(config['model_name_or_path'])
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss
        
    def encode(self, text):
        return self.main_model.encoder(**text).last_hidden_state

    def decode(self, encoder_outputs, attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return outputs
        #return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)