import copy
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch

from feat.base_featurizer import BaseFeaturizer
from feat.kg_featurizer import SUPPORTED_KG_FEATURIZER
from feat.text_featurizer import SUPPORTED_TEXT_FEATURIZER

from transformers import AutoTokenizer

class ProteinIndexFeaturizer(BaseFeaturizer):
    VOCAB_PROTEIN = { 
        "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
		"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
		"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
		"U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, 
	    "Z": 25
    }

    def __init__(self, config):
        super(ProteinIndexFeaturizer, self).__init__()
        self.max_length = config["max_len"]

    def __call__(self, data):
        temp = [self.VOCAB_PROTEIN[s] for s in data]
        if len(temp) < self.max_length:
            temp = np.pad(temp, (0, self.max_length - len(temp)))
        else:
            temp = temp[:self.max_length]
        return torch.LongTensor(temp)

class ProteinOneHotFeaturizer(BaseFeaturizer):
    amino_char = [
        '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
        'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z'
    ]
    
    def __init__(self, config):
        super(ProteinOneHotFeaturizer, self).__init__()
        self.max_length = config["max_len"]
        self.enc = OneHotEncoder().fit(np.array(self.amino_char).reshape(-1, 1))

    def __call__(self, data):
        temp = [i if i in self.amino_char else '?' for i in data]
        if len(temp) < self.max_length:
            temp = temp + ['?'] * (self.max_length - len(temp))
        else:
            temp = temp[:self.max_length]
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)

class ProteinTransformerTokFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(ProteinTransformerTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)

    def __call__(self, data):
        result = self.tokenizer(data, max_length=self.max_length, padding=True, truncation=True)
        return result

SUPPORTED_SINGLE_MODAL_PROTEIN_FEATURIZER = {
    "index": ProteinIndexFeaturizer,
    "OneHot": ProteinOneHotFeaturizer,
    "transformer": ProteinTransformerTokFeaturizer
}

class ProteinMultiModalFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(ProteinMultiModalFeaturizer, self).__init__()
        self.modality = config["modality"]
        self.featurizers = {}
        if "structure" in config["modality"]:
            conf = config["featurizer"]["structure"]
            self.featurizers["structure"] = SUPPORTED_SINGLE_MODAL_PROTEIN_FEATURIZER[conf["name"]](conf)
        if "kg" in config["modality"]:
            conf = config["featurizer"]["kg"]
            self.featurizers["kg"] = SUPPORTED_KG_FEATURIZER[conf["name"]](conf)
        if "text" in config["modality"]:
            conf = config["featurizer"]["text"]
            self.featurizers["text"] = SUPPORTED_TEXT_FEATURIZER[conf["name"]](conf)

    def set_protein2kgid_dict(self, protein2kgid):
        self.featurizers["kg"].set_transform(protein2kgid)

    def set_protein2text_dict(self, protein2text):
        self.featurizers["text"].set_transform(protein2text)

    def __call__(self, data):
        feat = {}
        for modality in self.featurizers.keys():
            feat[modality] = self.featurizers[modality](data)
        return feat

SUPPORTED_PROTEIN_FEATURIZER = copy.deepcopy(SUPPORTED_SINGLE_MODAL_PROTEIN_FEATURIZER)
SUPPORTED_PROTEIN_FEATURIZER["MultiModal"] = ProteinMultiModalFeaturizer