from abc import ABC, abstractmethod
from transformers import BertModel, BertTokenizer, T5Model, T5Tokenizer, GPT2Model, GPT2Tokenizer, LlamaModel, LlamaTokenizer

from open_biomed.feature.base_featurizer import BaseFeaturizer
from open_biomed.utils import ToDevice, get_biot5_tokenizer

# Warning: it seems that the results of AutoTokenizer and specified tokenizer is different
name2tokenizer = {
    "bert": BertTokenizer,
    "t5": T5Tokenizer,
    "gpt2": GPT2Tokenizer,
    "llama": LlamaTokenizer,
}
name2model = {
    "bert": BertModel,
    "t5": T5Model,
    "gpt2": GPT2Model,
    "llama": LlamaModel,
}

class TextFeaturizer(BaseFeaturizer, ABC):
    def __init__(self):
        super(TextFeaturizer, self).__init__()
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

class TextTransformerTokFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextTransformerTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.add_special_tokens = False if "no_special_tokens" in config else True
        if "prompt" in config:
            prompt = config["prompt"].split("<textHere>")
            self.prompt = prompt[0] + "{content}" + prompt[1]
        else:
            self.prompt = "{content}"
        if config["transformer_type"] == "biot5":
            self.tokenizer = get_biot5_tokenizer(config)
        else:
            self.tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)
        if config["transformer_type"] in ["gpt2"]:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if config["transformer_type"] == "llama":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_special_tokens({'bos_token': '</s>'})
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
            self.tokenizer.add_special_tokens({'unk_token': '</s>'})

    def __call__(self, data):
        if self.transform is not None:
            orig_data = data
            data = self.transform[data]
            if isinstance(data, list):
                if orig_data not in self.transform_count:
                    self.transform_count[orig_data] = 0
                data = data[self.transform_count[orig_data]]
                self.transform_count[orig_data] += 1
        return self.tokenizer(self.prompt.format(content=data), truncation=True, padding=True, add_special_tokens=self.add_special_tokens)

class TextTransformerSentFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextTransformerSentFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.min_sentence_length = config["min_sentence_length"]
        self.tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)
        if config["transformer_type"] in ["gpt2"]:
            self.tokenizer.pad_token = self.tokenizer.eos_token        

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        sents = []
        for sent in data.split("."):
            if len(sent.split(" ")) < 5:
                continue
            sents.append(self.tokenizer(sent, truncation=True, padding=True))
        return sents

class TextTransformerEncFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextTransformerEncFeaturizer, self).__init__()
        self.device = config["device"]

        self.max_length = config["max_length"]
        self.tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)
        if config["transformer_type"] in ["gpt2"]:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = name2model[config["transformer_type"]].from_pretrained(config["model_name_or_path"])
        self.encoder = self.encoder.to(self.device)
        
    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        data = self.tokenizer(data, truncation=True, padding=True, return_tensors='pt')
        data = ToDevice(data, self.device)
        return self.encoder(**data)["pooler_output"].detach().cpu()

class TextCLIPFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextCLIPFeaturizer, self).__init__()
        
    def __call__(self, data):
        import clip
        return clip.tokenize(data, truncate=True)

SUPPORTED_TEXT_FEATURIZER = {
    "TransformerTokenizer": TextTransformerTokFeaturizer,
    "TransformerSentenceTokenizer": TextTransformerSentFeaturizer,
    "TransformerEncoder": TextTransformerEncFeaturizer,
    "CLIP": TextCLIPFeaturizer,
}