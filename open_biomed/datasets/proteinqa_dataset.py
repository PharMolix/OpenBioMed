import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import numpy as np
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from open_biomed.feature.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from open_biomed.utils.mol_utils import valid_smiles

class ProteinQADataset(Dataset, ABC):
    def __init__(self, path, config):
        super(ProteinQADataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize(True if self.split == "train" else False)

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self, featurize_output=True):
        featurizer = SUPPORTED_PROTEIN_FEATURIZER[self.config["protein"]["featurizer"]["structure"]["name"]](self.config["protein"]["featurizer"]["structure"])
        self.proteins = [featurizer(prot) for prot in tqdm(self.proteins)]
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["question"]["name"]](self.config["text"]["question"])
        self.questions = [featurizer(text) for text in tqdm(self.questions)]
        if featurize_output:
            featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["answer"]["name"]](self.config["text"]["answer"])
            self.answers = [featurizer(text) for text in tqdm(self.answers)]

    def __getitem__(self, index):
        return [self.proteins[index]], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.questions)

class PQA(ProteinQADataset):
    def __init__(self, path, config, split):
        self.split = split
        super(PQA, self).__init__(path, config)

    def _load_data(self):
        data = json.load(open(os.path.join(self.path, self.split + ".json"), "r"))
        proteins_all = json.load(open(os.path.join(self.path, "sequences.json"), "r"))
        self.proteins = []
        self.sources = []
        self.protein_accs = []
        self.acc2idx = {}
        self.questions = []
        self.answers = []
        cnt = 0
        perm = np.random.permutation(len(data))
        for i in perm:
            """
            if len(proteins_all[sample["protein_accession"][0]]) > 400:
                continue
            """
            sample = data[i]
            self.sources.append(sample["source"])
            self.protein_accs.append(sample["protein_accession"])
            for prot in sample["protein_accession"]:
                if prot not in self.acc2idx:
                    self.acc2idx[prot] = cnt
                    cnt += 1
                    self.proteins.append(proteins_all[prot])
            self.questions.append(sample["question"])
            self.answers.append(sample["answer"])
        logger.info("Split: %s, # Proteins %d, # Questions: %d" % (self.split, len(self.proteins), len(self)))
        
    def __getitem__(self, index):
        return [self.proteins[self.acc2idx[acc]] for acc in self.protein_accs[index]], self.questions[index], self.answers[index]

SUPPORTED_PROTEINQA_DATASET = {
    "pqa": PQA,
}