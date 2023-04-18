import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

class BaseTransformers(nn.Module):
    def __init__(self, config):
        super(BaseTransformers, self).__init__()
        transformer_config = AutoConfig.from_pretrained(config["model_name_or_path"])
        if "load_model" in config:
            self.main_model = AutoModel.from_pretrained(config["model_name_or_path"])
        else:
            self.main_model = AutoModel(transformer_config)
            if "init_checkpoint" in config:
                ckpt = torch.load(config["init_checkpoint"])
                self.main_model.load_state_dict(ckpt)
        if "use_num_layers" in config:
            self.use_num_layers = config["use_num_layers"]
        else:
            self.use_num_layers = -1
        self.dropout = nn.Dropout(config["dropout"])
        self.pooler = config["pooler"]
        self.output_dim = transformer_config.hidden_size

    def forward(self, text):
        result = self.main_model(**text)
        if self.pooler == 'default':
            result = result['pooler_output']
        elif self.pooler == 'mean':
            result = torch.mean(result['last_hidden_state'], dim=-2)
        elif self.pooler == 'cls':
            result = result['last_hidden_state'][:, 0, :]
        return self.dropout(result)

    def encode_text(self, text, return_cls=False, proj=False):
        if self.use_num_layers == -1:
            result = self.main_model(**text)
            return result['last_hidden_state']
        else:
            text["output_hidden_states"] = True
            result = self.main_model(**text)
            return result['hidden_states'][self.use_num_layers]