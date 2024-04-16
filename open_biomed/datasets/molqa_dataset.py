import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import os
import csv
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from open_biomed.utils.mol_utils import valid_smiles

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
        self.mols = [featurizer(smi) for smi in tqdm(self.smiles)]
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["question"]["name"]](self.config["text"]["question"])
        self.questions = [featurizer(text) for text in tqdm(self.questions)]
        if featurize_output:
            featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["answer"]["name"]](self.config["text"]["answer"])
            self.answers = [featurizer(text) for text in tqdm(self.answers)]

    def __getitem__(self, index):
        return [self.mols[index]], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.questions)

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

class MQA(MolQADataset):
    def __init__(self, path, config, split):
        self.split = split
        super(MQA, self).__init__(path, config)

    def _load_data(self):
        data = json.load(open(os.path.join(self.path, self.split + ".json"), "r"))
        self.sources = []
        self.smiles = []
        self.smi2index = {}
        num_mols = 0
        self.smiles_indexes = []
        self.questions = []
        self.answers = []
        for sample in data:
            self.sources.append(sample["source"])
            self.smiles_indexes.append([])
            for smi in sample["smiles"]:
                if smi not in self.smi2index:
                    self.smiles.append(smi)
                    self.smi2index[smi] = num_mols
                    num_mols += 1
                self.smiles_indexes[-1].append(self.smi2index[smi])
            self.questions.append(sample["question"])
            self.answers.append(sample["answer"])
        logger.info("Split: %s, # Samples: %d" % (self.split, len(self)))
        
    def __getitem__(self, index):
        return [self.mols[i] for i in self.smiles_indexes[index]], self.questions[index], self.answers[index]

SUPPORTED_MOLQA_DATASET = {
    "chembl-qa": ChEMBLQA,
    "mqa": MQA,
}