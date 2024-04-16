import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

from open_biomed.models.base_models import TextEncoder

class BaseTransformers(TextEncoder):
    def __init__(self, config):
        super(BaseTransformers, self).__init__()
        transformer_config = AutoConfig.from_pretrained(config["model_name_or_path"])
        if "load_model" in config:
            self.main_model = AutoModel.from_pretrained(config["model_name_or_path"])
            #for name, param in self.main_model.named_parameters():
            #    print(name, param)
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

    def pool(self, h):
        if self.pooler == 'default':
            h = h['pooler_output']
        elif self.pooler == 'mean':
            h = torch.mean(h['last_hidden_state'], dim=-2)
        elif self.pooler == 'cls':
            h = h['last_hidden_state'][:, 0, :]
        return h

    def forward(self, text):
        result = self.main_model(**text)
        return self.dropout(result)

    def encode_text(self, text, pool=True, proj=False):
        if self.use_num_layers == -1:
            result = self.main_model(**text)
            if pool:
                h = self.pool(result)
                return self.dropout(h)
            else:
                return self.dropout(result['last_hidden_state'])
        else:
            text["output_hidden_states"] = True
            result = self.main_model(**text)
            return result['hidden_states'][self.use_num_layers]