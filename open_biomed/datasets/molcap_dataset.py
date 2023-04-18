import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import os
import csv

import torch
from torch.utils.data import Dataset

from feat.drug_featurizer import DrugMultiModalFeaturizer
from feat.text_featurizer import TextTransformerTokFeaturizer
from utils.mol_utils import valid_smiles

class MolCapDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MolCapDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        featurizer = DrugMultiModalFeaturizer(self.config)
        featurizer.set_drug2text_dict(self.smi2text)
        self.mols = [featurizer(smi) for smi in self.smiles]
        if "additional_text" in self.config["modality"]:
            featurizer = TextTransformerTokFeaturizer(self.config["featurizer"]["additional_text"])
            for i, smi in enumerate(self.smiles):
                self.mols[i]["additional_text"] = featurizer(self.smi2addtext[smi])

    def __getitem__(self, index):
        return self.mols[index]

    def __len__(self):
        return len(self.smiles)

class CheBI_20(MolCapDataset):
    def __init__(self, path, config, split):
        self.split = split
        super(CheBI_20, self).__init__(path, config)

    def _load_data(self):
        self.smiles = []
        self.texts = []
        with open(os.path.join(self.path, self.split + ".txt")) as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                if valid_smiles(line["SMILES"]):
                    self.smiles.append(line["SMILES"])
                    self.texts.append(line["description"])
                else:
                    logger.warn("Failed to generate 2D Graph for %s" % (line["SMILES"]))
        self.smi2text = dict(zip(self.smiles, self.texts))
        logger.info("Split: %s, Num Samples: %d" % (self.split, len(self)))
        if "additional_text" in self.config["modality"]:
            self.smi2addtext = {}
            with open(os.path.join(self.path, "molt5-captions-" + self.split + ".txt")) as f:
                reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                for line in reader:
                    self.smi2addtext[line["SMILES"]] = line["output"]

SUPPORTED_MOLCAP_DATASET = {
    "chebi-20": CheBI_20
}