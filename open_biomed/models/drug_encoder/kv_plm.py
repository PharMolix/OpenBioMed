import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from transformers import BertConfig, BertForPreTraining, BertModel

class KVPLMStarEncoder(nn.Module):
    def __init__(self, bert_config):
        super(KVPLMStarEncoder, self).__init__()
        self.ptmodel = BertForPreTraining(bert_config)
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids >= 30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i, j] = self.emb(input_ids[i, j] - 30700)
        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids)

class KVPLM(nn.Module):
    def __init__(self, config):
        super(KVPLM, self).__init__()

        bert_config = BertConfig.from_json_file(config["bert_config_path"])
        if config["name"] == "KV-PLM*":
            self.text_encoder = KVPLMStarEncoder(bert_config)
        else:
            self.text_encoder = BertModel(bert_config)
        ckpt = torch.load(config["checkpoint_path"])
        processed_ckpt = {}
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
            processed_ckpt = {k[7:]: v for k, v in ckpt.items()}
        elif 'bert.embeddings.word_embeddings.weight' in ckpt:
            for k, v in ckpt.items():
                if k.startswith("bert."):
                    processed_ckpt[k[5:]] = v
            
        missing_keys, unexpected_keys = self.text_encoder.load_state_dict(processed_ckpt, strict=False)
        logger.info("missing_keys: %s" % " ".join(missing_keys))
        logger.info("unexpected_keys: %s" % " ".join(unexpected_keys))

        self.dropout = nn.Dropout(config["dropout"])
        
        self.text_projector = nn.Sequential(
                nn.Linear(bert_config.hidden_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
            )

        self.output_dim = bert_config.hidden_size
        
    def forward(self, drug):
        return self.encode_structure(drug["strcture"]), self.encode_text(drug["text"])

    def encode_structure(self, structure):
        h = self.text_encoder(**structure)["pooler_output"]
        # h = self.text_projector(h)
        return self.dropout(h)

    def encode_text(self, text):
        h = self.text_encoder(**text)["pooler_output"]
        return self.dropout(h)