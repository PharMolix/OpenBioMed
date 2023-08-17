import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import os
import csv
import json

import torch
from torch.utils.data import Dataset

from feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER
from feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from utils.mol_utils import valid_smiles

class MolQADataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MolQADataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize(True if self.split == "train" else False)

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self, featurize_output=True):
        featurizer = SUPPORTED_MOL_FEATURIZER[self.config["mol"]["featurizer"]["structure"]["name"]](self.config["mol"]["featurizer"]["structure"])
        self.mols = [featurizer(smi) for smi in self.smiles]
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["question"]["featurizer"]["name"]](self.config["text"]["question"]["featurizer"])
        self.questions = [featurizer(text) for text in self.questions]
        if featurize_output:
            featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["answer"]["featurizer"]["name"]](self.config["text"]["answer"]["featurizer"])
            self.answers = [featurizer(text) for text in self.answers]

    def __getitem__(self, index):
        return self.mols[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.smiles)

class ChEMBLQA(MolQADataset):
    def __init__(self, path, config, split):
        self.split = split
        super(ChEMBLQA, self).__init__(path, config)

    def _load_data(self):
        self.smiles = []
        self.questions = []
        self.answers = []
        self.num_mols = 0
        data = json.load(open(os.path.join(self.path, "ChEMBL_QA_" + self.split + ".json"), "r"))
        for smi in data:
            if valid_smiles(smi):
                self.num_mols += 1
                for question in data[smi]:
                    self.smiles.append(smi)
                    self.questions.append(question[0])
                    self.answers.append(str(question[1]))
            else:
                logger.debug("Failed to generate 2D Graph for %s" % (smi))
        self.smi2text = dict(zip(self.smiles, self.questions))
        logger.info("Split: %s, Num Molecules: %d, Num Questions: %d" % (self.split, self.num_mols, len(self)))

class BioMedQA(MolQADataset):
    def __init__(self, path, config):
        super(BioMedQA).__init__(path, config)

SUPPORTED_MOLQA_DATASET = {
    "chembl-qa": ChEMBLQA 
}