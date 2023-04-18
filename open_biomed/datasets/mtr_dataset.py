"""
Dataset for Molecule-Text Retrieval
"""
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import os
import os.path as osp
import copy
import random

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np
import torch
from torch.utils.data import Dataset

from feat.drug_featurizer import DrugMultiModalFeaturizer
from utils.split import scaffold_split

class MTRDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MTRDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        # featurize drug with paired text
        featurizer = DrugMultiModalFeaturizer(self.config["drug"])
        featurizer.set_drug2text_dict(self.drug2text)
        self.drugs = [featurizer(drug) for drug in self.drugs]

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        return new_dataset

    def __getitem__(self, index):
        return self.drugs[index]

    def __len__(self):
        return len(self.drugs)

class PCdes(MTRDataset):
    def __init__(self, path, config, mode='paragraph', filter=True, filter_path=""):
        self.filter = filter
        self.filter_path = filter_path
        self.test = False
        self.mode = mode
        super(PCdes, self).__init__(path, config)
        self._train_test_split()

    def _load_data(self):
        with open(osp.join(self.path, "align_smiles.txt"), "r") as f:
            drugs = f.readlines()

        with open(osp.join(self.path, "align_des_filt3.txt"), "r") as f:
            texts = f.readlines()[:len(drugs)]

        if self.filter:
            with open(self.filter_path, "r") as f:
                filter_drugs = []
                for line in f.readlines():
                    drug = line.rstrip("\n").split("\t")[1]
                    mol = Chem.MolFromSmiles(drug)
                    if mol is not None:
                        filter_drugs.append(Chem.MolToSmiles(mol, isomericSmiles=True))

        self.drugs = []
        self.texts = []
        for i, drug in enumerate(drugs):
            try:
                mol = Chem.MolFromSmiles(drug.strip("\n"))
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                if mol is not None and not smi in filter_drugs:
                    self.drugs.append(smi)
                    self.texts.append(texts[i].strip("\n"))
            except:
                logger.warn("fail to generate 2D graph, data removed")

        self.smiles = self.drugs
        self.drug2text = dict(zip(self.drugs, self.texts))
        logger.info("Num Samples: %d" % len(self))

    def _train_test_split(self):
        self.train_index, self.val_index, self.test_index = scaffold_split(self, 0.1, 0.2)

    def set_test(self):
        self.test = True
        if self.mode == "sentence":
            rnd = 42
            self.pseudorandom = []
            for i in range(len(self)):
                self.pseudorandom.append(rnd)
                rnd = rnd * 16807 % ((1 << 31) - 1)

    def __getitem__(self, index):
        if self.mode == "sentence":
            if not self.test:
                ind = random.randint(0, len(self.drugs[index]["text"]) - 1)
            else:
                ind = self.pseudorandom[index] % len(self.drugs[index]["text"])
            return {
                "structure": self.drugs[index]["structure"],
                "text": self.drugs[index]["text"][ind],
            }
        else:
            return self.drugs[index]

class PubChem15K(MTRDataset):
    def __init__(self, path, config):
        super(PubChem15K, self).__init__(path, config)
        self._train_test_split()

    def _load_data(self):
        random.seed(42)
        self.drugs, self.texts = [], []
        with open(os.path.join(self.path, "pair.txt")) as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                text_name, smi = line[0], line[1]
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        self.drugs.append(smi)
                        text_list = []
                        count = 0
                        for line in open(os.path.join(self.path, "text", "text_" + text_name + ".txt"), 'r', encoding='utf-8'):
                            count += 1
                            text_list.append(line)
                            if count > 500:
                                break
                        #text = random.sample(text_list, 1)[0]
                        text = text_list[0]
                        if len(text) > 256:
                            text = text[:256]
                        self.texts.append(text)
                        print(smi, text)
                except:
                    continue
                if len(self.drugs) >= 480:
                    break
        self.drug2text = dict(zip(self.drugs, self.texts))

    def _train_test_split(self):
        self.train_index = np.arange(0, 480)
        self.val_index = np.arange(0, 480)
        self.test_index = np.arange(0, 480)

SUPPORTED_MTR_DATASETS = {
    "PCdes": PCdes,
    "PubChem15K": PubChem15K,
}