from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

from collections import OrderedDict
import csv
import copy
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer
from open_biomed.feature.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from open_biomed.utils.mol_utils import can_smiles
from open_biomed.utils.kg_utils import SUPPORTED_KG, embed
from open_biomed.utils.split_utils import kfold_split, cold_drug_split, cold_protein_split, cold_cluster_split, cold_mssl2drug_split, random_split


class DDIDataset(Dataset, ABC):
    def __init__(self, path, config, split_strategy, in_memory=True):
        super(DDIDataset, self).__init__()
        self.path = path
        self.config = config
        self.split_strategy = split_strategy
        self.in_memory = in_memory
        self._load_data()
        self._train_test_split()
        self.kge = None

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _train_test_split(self):
        raise NotImplementedError

    def _build(self, eval_pair_index, save_path):
        # build after train / test datasets are determined for filtering out edges
        if len(self.config["drug"]["modality"]) > 1:
            kg_config = self.config["drug"]["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            self.drug2kg, self.drug2text, _, _ = self.kg.link(self)
            self.concat_text_first = self.config["concat_text_first"]
            filter_out = []
            for i_drugA, i_drugB in eval_pair_index:
                smiA = self.smiles[i_drugA]
                smiB = self.smiles[i_drugB]
                if smiA in self.drug2kg and smiB in self.drug2kg:
                    filter_out.append((self.drug2kg[smiA], self.drug2kg[smiB]))
            # embed once for consistency
            kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], save=True,
                        save_path=save_path)
            self.kge = kge
            self.config["drug"]["featurizer"]["kg"]["kge"] = kge
        else:
            self.concat_text_first = False
        self._configure_featurizer()
        # featurize all data pairs in one pass for training efficiency
        if self.in_memory:
            self._featurize()

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.pair_index = [self.pair_index[i] for i in indexes]
        new_dataset.labels = [self.labels[i] for i in indexes]
        return new_dataset

    def split_train_val(self, Nval):
        val_dataset = copy.deepcopy(self)
        val_idx_prelim = random.sample(range(len(self.pair_index)), Nval)
        train_idx_prelim = list(set(range(len(self.pair_index))) - set(val_idx_prelim))
        train_pair_prelim = [self.pair_index[idx] for idx in train_idx_prelim]
        to_val = []
        to_train = []
        for i, idx in enumerate(val_idx_prelim):
            val_pair = self.pair_index[idx]
            try:
                train_i = train_pair_prelim.index((val_pair[1], val_pair[0]))
            except ValueError:
                pass
            else:
                if i % 2 == 0: to_train.append(idx)
                else: to_val.append(train_idx_prelim[train_i])

        val_idx = set(val_idx_prelim).difference(set(to_train)).union(set(to_val))
        train_idx = set(train_idx_prelim).difference(set(to_val)).union(set(to_train))
        val_idx = list(val_idx)
        train_idx = list(train_idx)
        val_idx.sort()
        train_idx.sort()

        self.pair_index = [self.pair_index[idx] for idx in train_idx]
        self.labels = [self.labels[idx] for idx in train_idx]
        self.featurized_drugsA = [self.featurized_drugsA[idx] for idx in train_idx]
        self.featurized_drugsB = [self.featurized_drugsB[idx] for idx in train_idx]
        val_dataset.pair_index = [val_dataset.pair_index[idx] for idx in val_idx]
        val_dataset.labels = [val_dataset.labels[idx] for idx in val_idx]
        val_dataset.featurized_drugsA = [val_dataset.featurized_drugsA[idx] for idx in val_idx]
        val_dataset.featurized_drugsB = [val_dataset.featurized_drugsB[idx] for idx in val_idx]

        return self, val_dataset

    def _configure_featurizer(self):
        if len(self.config["drug"]["modality"]) > 1:
            self.drug_featurizer = MolMultiModalFeaturizer(self.config["drug"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            if not self.concat_text_first:
                self.drug_featurizer.set_drug2text_dict(self.drug2text)
        else:
            drug_feat_config = self.config["drug"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_MOL_FEATURIZER[drug_feat_config["name"]](drug_feat_config)

    def _featurize(self):
        logger.info("Featurizing...")
        self.featurized_drugsA = []
        self.featurized_drugsB = []
        for i_drugA, i_drugB in tqdm(self.pair_index):
            drugA, drugB = self.smiles[i_drugA], self.smiles[i_drugB]
            if len(self.config["drug"]["modality"]) > 1 and self.concat_text_first:
                processed_drugA = self.drug_featurizer(drugA, skip=["text"])
                processed_drugB = self.drug_featurizer(drugB)
                processed_drugA["text"] = self.drug_featurizer["text"](
                    self.drug2text[drugA] + " [SEP] " + self.drug2text[drugB]
                )
            else:
                processed_drugA = self.drug_featurizer(drugA)
                processed_drugB = self.drug_featurizer(drugB)
            self.featurized_drugsA.append(processed_drugA)
            self.featurized_drugsB.append(processed_drugB)

    def __getitem__(self, index):
        if not self.in_memory:
            drugA, drugB, label = \
                self.smiles[self.pair_index[index][0]], \
                self.smiles[self.pair_index[index][1]], \
                self.labels[index]
            processed_drugA = self.drug_featurizer(drugA)
            processed_drugB = self.drug_featurizer(drugB)
            if self.concat_text_first:
                processed_drugA["text"] = self.drug_featurizer["text"](
                    self.drug2text[drugA] + " [SEP] " + self.drug2text[drugB])
            return processed_drugA, processed_drugB, label
        else:
            return self.featurized_drugsA[index], self.featurized_drugsB[index], self.labels[index]

    def __len__(self):
        return len(self.pair_index)

class MSSL2drug(DDIDataset):
    def __init__(self, path, config, split_strategy):
        super(MSSL2drug, self).__init__(path, config, split_strategy)

    def _load_data(self):
        with open(os.path.join(self.path, 'drug.txt'), 'r') as f:
            data = [line for line in f.read().split('\n') if line != '']
        self.drug_keys = [line[:7] for line in data]
        drug_keys2name = {line[:7] : line[8:] for line in data}
        with open(os.path.join(self.path, 'drug_smiles.json'), 'r') as f:
            drug_name2smiles = json.load(f)
        self.smiles = [drug_name2smiles[drug_keys2name[key]] for key in self.drug_keys]
        self.drugid2index = dict(zip(self.drug_keys, range(len(self.smiles))))

        with open(os.path.join(self.path, 'DDInet.txt'), 'r') as f:
            data = [line.split(' ') for line in f.read().split('\n') if line != '']
        assert len(data) == len(self.drug_keys)

        self.pair_index, self.labels = [], []
        for i in range(len(data)):
            row = data[i]
            assert len(row) == len(self.drug_keys)
            for j in range(i+1):
                label = int(row[j])
                self.pair_index.append((i, j))
                self.labels.append(label)

    def _train_test_split(self):
        self.nfolds = 5
        if self.split_strategy == "warm":
            folds = kfold_split(len(self), self.nfolds)
            self.folds = []
            print('Filtering out contamination pairs in training set...')
            for i in range(self.nfolds):
                train_prelim = np.concatenate(folds[:i] + folds[i + 1:], axis=0).tolist()
                test = folds[i]
                test_pair_index = [self.pair_index[idx] for idx in test]
                train = []
                for idx in tqdm(train_prelim):
                    pair_index = self.pair_index[idx]
                    if (pair_index[1], pair_index[0]) not in test_pair_index:
                        train.append(idx)
                self.folds.append({
                    "train": train,
                    "test": test
                })
        elif self.split_strategy == 'cold':
            self.folds = cold_mssl2drug_split(self, self.nfolds)
        else:
            raise RuntimeError('Unsupported split strategy:', self.split_strategy)

class DrugBank(DDIDataset):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()
        self._train_test_split()
        
    def _load_data(self):
        drugs = json.load(open(os.path.join(self.path, "drug_set.json"), "r"))
        self.mols = []
        dbkey2id = {}
        for key in drugs:
            self.mols.append(drugs[key]["smiles"])
            dbkey2id[key] = len(self.mols) - 1
        self.pair_index = []
        self.labels = []
        with open(os.path.join(self.path, "labeled_triples.csv")) as f:
            reader = csv.DictReader(f, delimiter=",")
            for line in reader:
                if float(line["label"]) == 1:
                    idx = int(line["context"][-2:]) - 1
                    self.pair_index.append((dbkey2id[line["drug_1"]], dbkey2id[line["drug_2"]]))
                    label = [0.0] * 86
                    label[idx] = 1
                    self.labels.append(label)
        print("Total ", len(self), " samples")

    def _featurize(self):
        featurizer = SUPPORTED_MOL_FEATURIZER[self.config["featurizer"]["structure"]["name"]](self.config["featurizer"]["structure"])
        self.mols = [featurizer(mol) for mol in self.mols]

    def _train_test_split(self):
        self.train_index, self.val_index, self.test_index = random_split(len(self), 0.2, 0.2)
    
    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.pair_index = [new_dataset.pair_index[i] for i in indexes]
        new_dataset.labels = [new_dataset.labels[i] for i in indexes]
        return new_dataset

    def __getitem__(self, index):
        return self.mols[self.pair_index[index][0]], self.mols[self.pair_index[index][1]], self.labels[index]

    def __len__(self):
        return len(self.labels)

SUPPORTED_DDI_DATASETS = {
    'mssl2drug': MSSL2drug,
    "drugbank": DrugBank
}