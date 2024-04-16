import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from transformers import BertConfig, BertForPreTraining, BertModel

from open_biomed.models.base_models import MolEncoder, TextEncoder

class KVPLMStarEncoder(nn.Module):
    def __init__(self, bert_config):
        super(KVPLMStarEncoder, self).__init__()
        self.ptmodel = BertForPreTraining(bert_config)
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids, output_hidden_states=False):
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids >= 30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i, j] = self.emb(input_ids[i, j] - 30700)
        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)

class KVPLM(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(KVPLM, self).__init__()

        bert_config = BertConfig.from_json_file(config["bert_config_path"])
        if config["name"] == "kv-plm*":
            self.text_encoder = KVPLMStarEncoder(bert_config)
        else:
            self.text_encoder = BertModel(bert_config)
        self.use_num_layers = config["use_num_layers"] if "use_num_layers" in config else -1
        ckpt = torch.load(config["init_checkpoint"])
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
        
    def forward(self, drug):
        return self.encode_mol(drug["strcture"]), self.encode_text(drug["text"])

    def encode_mol(self, structure):
        h = self.text_encoder(**structure)["pooler_output"]
        return self.dropout(h)

    def encode_text(self, text, return_cls=False, proj=False):
        if self.use_num_layers != -1:
            text["output_hidden_states"] = True
        output = self.text_encoder(**text)
        if return_cls:
            logits = output["pooler_output"]
            logits = self.dropout(logits)
        elif self.use_num_layers == -1:
            logits = output["last_hidden_state"]
        else:
            logits = output["hidden_states"][self.use_num_layers]
        return logits