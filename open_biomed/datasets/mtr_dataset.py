"""
Dataset for Molecule-Text Retrieval
"""
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import os
import os.path as osp
import copy
import json
import random
from tqdm import tqdm

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np
import torch
from torch.utils.data import Dataset

from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from open_biomed.utils.split_utils import scaffold_split

class MTRDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MTRDataset, self).__init__()
        self.path = path
        self.config = config
        self.is_multimodal = len(self.config["mol"]["modality"]) > 1
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        # featurize mol with paired text
        if self.is_multimodal:
            mol_featurizer = MolMultiModalFeaturizer(self.config["mol"])
            mol_featurizer.set_mol2text_dict(self.mol2text)
        else:
            mol_featurizer = SUPPORTED_MOL_FEATURIZER[self.config["mol"]["featurizer"]["structure"]["name"]](self.config["mol"]["featurizer"]["structure"])
            
        self.mols = [mol_featurizer(mol) for mol in tqdm(self.mols)]
        text_featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["name"]](self.config["text"])
        self.texts = [text_featurizer(text) for text in tqdm(self.texts)]

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.mols = [new_dataset.mols[i] for i in indexes]
        new_dataset.texts = [new_dataset.texts[i] for i in indexes]
        return new_dataset

    def __len__(self):
        return len(self.mols)

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
                ind = random.randint(0, len(self.mols[index]["text"]) - 1)
            else:
                ind = self.pseudorandom[index] % len(self.texts["text"])
            return self.mols[index], self.texts[index][ind],
        else:
            return self.mols[index], self.texts[index]

class PCdes(MTRDataset):
    def __init__(self, path, config, mode='paragraph', perspective="None", filter=True, filter_path=""):
        self.filter = filter
        self.filter_path = filter_path
        self.test = False
        self.mode = mode
        super(PCdes, self).__init__(path, config)
        self._train_test_split()

    def _load_data(self):
        with open(osp.join(self.path, "align_smiles.txt"), "r") as f:
            mols = f.readlines()

        with open(osp.join(self.path, "align_des_filt3.txt"), "r") as f:
            texts = f.readlines()[:len(mols)]

        if self.filter:
            with open(self.filter_path, "r") as f:
                filter_mols = []
                for line in f.readlines():
                    mol = line.rstrip("\n").split("\t")[1]
                    mol = Chem.MolFromSmiles(mol)
                    if mol is not None:
                        filter_mols.append(Chem.MolToSmiles(mol, isomericSmiles=True))

        self.mols = []
        self.texts = []
        for i, mol in enumerate(mols):
            try:
                mol = Chem.MolFromSmiles(mol.strip("\n"))
                smi_orig = Chem.MolToSmiles(mol, isomericSmiles=False)
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                if mol is not None and not smi in filter_mols:
                    self.mols.append(smi_orig)
                    self.texts.append(texts[i].strip("\n"))
            except:
                logger.debug("fail to generate 2D graph, data removed")
        self.mol2text = dict(zip(self.mols, ["View: chemical properties and functions" for i in range(len(self.mols))]))
        self.smiles = self.mols
        logger.info("Num Samples: %d" % len(self))

    def _train_test_split(self):
        self.train_index, self.val_index, self.test_index = scaffold_split(self, 0.1, 0.2)

class PubChem15K(MTRDataset):
    def __init__(self, path, config, mode, perspective, filter, filter_path):
        self.mode = mode
        self.test = False
        super(PubChem15K, self).__init__(path, config)
        self._train_test_split()

    def _load_data(self):
        random.seed(42)
        self.mols, self.texts = [], []
        with open(os.path.join(self.path, "pair.txt")) as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                text_name, smi = line[0], line[1]
                try:
                    mol = Chem.MolFromSmiles(smi)
                    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                    if mol is not None:
                        self.mols.append(smi)
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
                except:
                    continue
                if len(self.mols) >= 480:
                    break
        self.mol2text = dict(zip(self.mols, self.texts))

    def _train_test_split(self):
        self.train_index = np.arange(0, 480)
        self.val_index = np.arange(0, 480)
        self.test_index = np.arange(0, 480)

class MPRetr(MTRDataset):
    def __init__(self, path, config, mode, perspective, filter, filter_path):
        if perspective == "mix":
            self.perspective = ["chemical properties and functions", "physical properties", "pharmacokinetic properties"]
        else:
            self.perspective = [" ".join(perspective.split("_"))]
        self.mode = mode
        super(MPRetr, self).__init__(path, config)
        
    def _load_data(self):
        self.train_index, self.val_index, self.test_index = [], [], []
        self.mols, self.texts, self.prompts = [], [], []
        cnt = 0
        for split in ["train", "val", "test"]:
            data = json.load(open(os.path.join(self.path, split + ".json"), "r"))
            for perspective in self.perspective:
                print(perspective, len(data[perspective]))
                for sample in data[perspective]:
                    try:
                        mol = Chem.MolFromSmiles(sample[0].strip("\n"))
                        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                        if mol is not None:
                            self.mols.append(smi)
                            self.texts.append(sample[1].strip("\n"))
                            self.prompts.append("View: " + perspective)
                            #self.prompts.append("Not Available")
                            if split == "train":
                                self.train_index.append(cnt)
                            if split == "val":
                                self.val_index.append(cnt)
                            if split == "test":
                                self.test_index.append(cnt)
                            cnt += 1
                    except:
                        logger.debug("fail to generate 2D graph, data removed")
        #print(len(self.train_index), len(self.val_index), len(self.test_index))
        #print(len(self.mols), len(self.prompts))
        self.mol2text = {}
        for i in range(len(self.mols)):
            if self.mols[i] not in self.mol2text:
                self.mol2text[self.mols[i]] = [self.prompts[i]]
            else:
                self.mol2text[self.mols[i]].append(self.prompts[i])
        # print(self.mol2text)

SUPPORTED_MTR_DATASETS = {
    "PCdes": PCdes,
    "PubChem15K": PubChem15K,
    "MPRetr": MPRetr,
}