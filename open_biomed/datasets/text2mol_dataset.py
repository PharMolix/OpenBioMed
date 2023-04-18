from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from feat.text_featurizer import SUPPORTED_TEXT_FEATURIZER

class Text2MolGenDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(Text2MolGenDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        feat_config = self.config["featurizer"]["text"]
        featurizer = SUPPORTED_TEXT_FEATURIZER[feat_config["name"]](feat_config)
        self.texts = [featurizer(text) for text in self.texts]

class PromptGenDataset(Text2MolGenDataset):
    def __init__(self, path, config):
        super().__init__(path, config)

    def _load_data(self):
        self.texts = [
            'The molecule is soluable in water.',
            'The molecule is not soluable in water.',
            'The molecule has high permeability.',
            'The molecule has low permeability.',
            'The molecule is like Felodipine.',
            'The molecule is like Lercanidipine.'
        ]
        """
        self.texts = [
            'The molecule is beautiful.', 
            'The molecule is versatile.', 
            'The molecule is strange.',
            'fluorescent molecules', 
            'The molecule contains hydroxyl and carboxyl groups, which can be thermally decomposed to generate ammonia gas, and the oxygen content in the molecule is not less than 20%.',
            'The molecule has high water solubility and barrier permeability with low toxicity.',
            'molecules containing nucleophilic groups',
            'molecules containing electrophilic groups',
            'molecules containing hydrophilic groups', 
            'molecules containing lipophilic groups'
        ]
        """
        self.smiles = None

class CheBI20(Text2MolGenDataset):
    def __init__(self, path, config):
        super().__init__(path, config)

    def _load_data(self):
        pass

SUPPORTED_TEXT2MOLGEN_DATASET = {
    "prompt": PromptGenDataset,
    "CheBI-20": CheBI20
}