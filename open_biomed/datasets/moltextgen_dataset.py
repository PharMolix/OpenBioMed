import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import os
import copy
import csv
import json
from rdkit import Chem
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from open_biomed.utils.mol_utils import valid_smiles

class MolTextGenDataset(Dataset):
    def __init__(self, config):
        super(MolTextGenDataset, self).__init__()
        self.config = config

    @staticmethod
    def from_list(config, smiles, texts):
        dataset = MolTextGenDataset(config)
        dataset.smiles = smiles
        dataset.texts = texts
        dataset.prompts = ["chemical properties and functions"] * len(dataset)
        dataset.smi2text = dict(zip(dataset.smiles, dataset.prompts))
        dataset._featurize()
        return dataset

    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        if len(self.config["mol"]["modality"]) > 1:
            flag = None
            if "conformation" in self.config["mol"]["featurizer"]["structure"]:
                flag = "conformation"
            if "unimol" in self.config["mol"]["featurizer"]["structure"]:
                flag = "unimol"
            if flag is not None:
                self.config["mol"]["featurizer"]["structure"][flag]["cache_file"] = "_".join(self.config["mol"]["featurizer"]["structure"][flag]["cache_file"].split("_")[:-1]) + "_" + self.split + ".pkl"
                logger.info("Cache to %s" % self.config["mol"]["featurizer"]["structure"][flag]["cache_file"])
            featurizer = MolMultiModalFeaturizer(self.config["mol"])
            featurizer.set_mol2text_dict(self.smi2text)
        else:
            featurizer = SUPPORTED_MOL_FEATURIZER[self.config["mol"]["featurizer"]["structure"]["name"]](self.config["mol"]["featurizer"]["structure"])
        self.mols = [featurizer(smi) for smi in tqdm(self.smiles)]
        """
        import pickle
        print(len(featurizer.featurizers["conformation"].cached_data), featurizer.featurizers["conformation"].cache_file)
        pickle.dump(featurizer.featurizers["conformation"].cached_data, open(featurizer.featurizers["conformation"].cache_file, "wb"))
        """
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["name"]](self.config["text"])
        self.texts_raw = copy.deepcopy(self.texts)
        self.texts = [featurizer(text) for text in self.texts]

    def __getitem__(self, index):
        return self.mols[index], self.texts[index]

    def __len__(self):
        return len(self.smiles)

class CheBI20(MolTextGenDataset):
    def __init__(self, path, config, split):
        super(CheBI20, self).__init__(config)
        self.split = split
        self.path = path
        self._load_data()
        self._featurize()

    def _load_data(self):
        self.smiles = []
        self.texts = []
        self.prompts = []
        with open(os.path.join(self.path, self.split + ".txt")) as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                if valid_smiles(line["SMILES"]):
                    self.smiles.append(line["SMILES"])
                    self.texts.append(line["description"])
                    self.prompts.append("chemical properties and functions")
                else:
                    logger.warn("Failed to generate 2D Graph for %s" % (line["SMILES"]))
        self.smi2text = dict(zip(self.smiles, self.prompts))
        logger.info("Split: %s, Num Samples: %d" % (self.split, len(self)))

class CheBIDia(MolTextGenDataset):
    def __init__(self, path, config, split):
        assert split == "train"
        super(CheBIDia, self).__init__(config)
        self.split = split
        self.path = path
        self._load_data()
        self._featurize()

    def _load_data(self):
        self.prompts = []
        self.texts = open(os.path.join(self.path, self.split + "_inp.txt"), "r").readlines()
        if self.config["mol"]["featurizer"]["structure"]["name"] == "selfies":
            self.texts = [self._replace_smiles_with_selfies(text) for text in self.texts]
        self.smiles = open(os.path.join(self.path, self.split + "_out.txt"), "r").readlines()
        self.prompts = ["chemical properties and functions"] * len(self.smiles)
        self.smi2text = dict(zip(self.smiles, self.prompts))
        logger.info("Split: %s, Num Samples: %d" % (self.split, len(self)))

    def _replace_smiles_with_selfies(self, text):
        text = text.split(" ")
        for i in range(1, len(text)):
            if i >= 3 and text[i - 3] == "It" and text[i - 2] == "looks" and text[i - 1] == "like" or text[i - 1] == "Modifying":
                import selfies as sf
                text[i] = Chem.MolToSmiles(Chem.MolFromSmiles(text[i].rstrip(".\n")))
                text[i] = sf.encoder(text[i], strict=False)
        return " ".join(text)

class MultiRoundMolTextGenDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MultiRoundMolTextGenDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def prepare_round(self, round, outputs):
        cur = 0
        new_smiles = []
        new_texts = []
        for i in range(len(self)):
            if len(self.texts[i]) > round:
                new_smiles.append(self.smiles[i][round])
                if round > 0:
                    new_texts.append(self.texts[i][round] + "It looks like " + outputs[cur] + ".")
                else:
                    new_texts.append(self.texts[i][round])
            if len(self.texts[i]) >= round:
                cur += 1
        assert round == 0 or cur == len(outputs)
        # print(new_smiles, new_texts)
        return MolTextGenDataset.from_list(self.config, new_smiles, new_texts)

class CheBIDiaTest(MultiRoundMolTextGenDataset):
    def __init__(self, path, config, split):
        assert split in ["validation", "test"]
        self.split = split
        super(CheBIDiaTest, self).__init__(path, config)

    def _load_data(self):
        self.smiles = []
        self.texts = []
        self.max_rounds = 0
        with open(os.path.join(self.path, self.split + "_inp.txt"), "r") as f:
            for line in f.readlines():
                self.texts.append(line.rstrip("\n").split("\t")[:-1])
                if len(self.texts[-1]) > self.max_rounds:
                    self.max_rounds = len(self.texts[-1])

        with open(os.path.join(self.path, self.split + "_out.txt"), "r") as f:
            for i, line in enumerate(f.readlines()):
                self.smiles.append(line.rstrip("\n").split("\t")[:-1])
                assert(len(self.smiles[i]) == len(self.texts[i]))

        logger.info("Num samples: %d, Max rounds: %d" % (len(self), self.max_rounds))

    def __len__(self):
        return len(self.smiles)

SUPPORTED_MOLCAP_DATASET = {
    "chebi-20": CheBI20,
}

SUPPORTED_TEXT2MOLGEN_DATASET = {
    "chebi-20": CheBI20,
    "chebi-dia": CheBIDia,
    "chebi-dia-test": CheBIDiaTest
}