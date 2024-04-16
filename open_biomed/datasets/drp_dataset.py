from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import copy
import pickle
import json
import random
import numpy as np
import pandas as pd
import os.path as osp

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer
from open_biomed.feature.cell_featurizer import SUPPORTED_CELL_FEATURIZER
from open_biomed.utils.cell_utils import SUPPORTED_GENE_SELECTOR

class DRPDataset(Dataset, ABC):
    def __init__(self, path, config, task="regression"):
        super(DRPDataset, self).__init__()
        self.path = path
        self.config = config
        self.task = task
        self.gene_selector = SUPPORTED_GENE_SELECTOR[config["cell"]["gene_selector"]]()
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        # featurize drug
        if len(self.config["mol"]["modality"]) > 1:
            featurizer = MolMultiModalFeaturizer(self.config["mol"])
        else:
            featurizer = SUPPORTED_MOL_FEATURIZER[self.config["mol"]["featurizer"]["structure"]["name"]](self.config["mol"]["featurizer"]["structure"])
        for key in self.drug_dict:
            smi = self.drug_dict[key]
            self.drug_dict[key] = featurizer(smi)
        
        # featurize cell
        featurizer = SUPPORTED_CELL_FEATURIZER[self.config["cell"]["featurizer"]["name"]](self.config["cell"]["featurizer"])
        self.cell_dict = featurizer(self.cell_dict)
        if self.config["cell"]["featurizer"]["name"] == "TGSA":
            self.predefined_cluster = featurizer.predefined_cluster

        # convert labels to tensor
        if self.task == "regression":
            self.IC = torch.FloatTensor(self.IC)
        if self.task == "classification":
            self.response = torch.FloatTensor(self.response)

    def _train_test_split(self):
        N = len(self)
        if self.config["split"]["type"] == "random":
            train_ratio, val_ratio = self.config["split"]["train"], self.config["split"]["val"]
            indexes = np.random.permutation(N)
            self.train_indexes = indexes[:int(N * train_ratio)]
            self.val_indexes = indexes[int(N * train_ratio): int(N * (train_ratio + val_ratio))]
            self.test_indexes = indexes[int(N * (train_ratio + val_ratio)):]
        elif self.config["split"]["type"] == "kfold":
            nfolds = self.config["split"]["nfolds"]
            indexes = np.random.permutation(N)
            self.fold_indexes = [indexes[int(N * k / nfolds): int(N * (k + 1) / nfolds)] for k in range(nfolds)]
        elif self.config["split"]["type"] == "cell":
            train_ratio, val_ratio = self.config["split"]["train"], self.config["split"]["val"]
            cells = list(set(self.cell_dict.keys()))
            random.shuffle(cells)
            cell_num = len(cells)
            train_cells = cells[:int(cell_num * train_ratio)]
            val_cells = cells[int(cell_num * train_ratio): int(cell_num * (train_ratio + val_ratio))]
            test_cells = cells[int(cell_num * (train_ratio + val_ratio)):]
            self.train_indexes, self.val_indexes, self.test_indexes = [], [], []
            for i in range(N):
                if self.cell_index[i] in val_cells:
                    self.val_indexes.append(i)
                elif self.cell_index[i] in test_cells:
                    self.test_indexes.append(i)
                else:
                    self.train_indexes.append(i)
            self.train_indexes, self.val_indexes, self.test_indexes = np.array(self.train_indexes), np.array(self.val_indexes), np.array(self.test_indexes)


    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drug_index = new_dataset.drug_index[indexes]
        new_dataset.cell_index = new_dataset.cell_index[indexes]
        if self.task == "regression":
            new_dataset.IC = new_dataset.IC[indexes]
        if self.task == "classification":
            new_dataset.response = new_dataset.response[indexes]
        return new_dataset

    def __getitem__(self, index):
        return self.drug_dict[self.drug_index[index]], self.cell_dict[self.cell_index[index]], self.IC[index] if self.task == "regression" else self.response[index]

    def __len__(self):
        return len(self.IC) if self.task == "regression" else len(self.response)

class GDSC(DRPDataset):
    def __init__(self, path, config, task="regression"):
        super(GDSC, self).__init__(path, config, task)
        self._train_test_split()
        
    def _load_data(self):
        # load drug information
        data_drug = np.loadtxt(osp.join(self.path, "GDSC_MolAnnotation.csv"), dtype=str, delimiter=',', comments='?', skiprows=1)
        self.drug_dict = dict([
            (data_drug[i][0], data_drug[i][2]) for i in range(data_drug.shape[0])
        ])
        
        # load cell-line information
        save_path = osp.join(self.path, "celldict_%s_%s.pkl" % ("-".join(self.config["cell"]["gene_feature"]), "-".join([str(v) for v in self.config["cell"]["featurizer"].values()])))
        if osp.exists(save_path):
            self.cell_dict = pickle.load(open(save_path, "rb"))
        else:
            self.cell_dict = {}
            for feat in self.config["cell"]["gene_feature"]:
                data_cell = np.loadtxt(osp.join(self.path, "GDSC_%s.csv" % (feat)), dtype=str, delimiter=',')
                gene_names = data_cell[0][1:]
                # select genes strongly related to tumor expression
                selected_idx = [0] + self.gene_selector(gene_names)
                data_cell = data_cell[1:, selected_idx]
                for cell in data_cell:
                    if cell[0] not in self.cell_dict:
                        self.cell_dict[cell[0]] = cell[1:].reshape(-1, 1).astype(np.float)
                    else:
                        self.cell_dict[cell[0]] = np.concatenate((self.cell_dict[cell[0]], cell[1:].reshape(-1, 1).astype(np.float)), axis=1)
                
            pickle.dump(self.cell_dict, open(save_path, "wb"))

        # load drug-cell response information
        data_IC50 = pd.read_csv(osp.join(self.path, "GDSC_DR.csv"))
        self.drug_index = data_IC50["MOL_NAME"].to_numpy()
        self.cell_index = data_IC50["cell_tissue"].to_numpy()
        self.IC = data_IC50["LN_IC50"].astype(np.float)
        resp2val = {'R': 1, 'S': 0}
        self.response = np.array([resp2val[x] for x in data_IC50["BinaryResponse"]])

class TCGA(DRPDataset):
    def __init__(self, path, config, subset="BRCA_28"):
        self.subset = subset
        super(TCGA, self).__init__(path, config, task="classification")

    def _load_data(self):
        # load cell-line data
        feat2file = {"EXP": "xena_gex", "MUT": "xena_mutation"}
        save_path = osp.join(self.path, "celldict_%s_%s.pkl" % ("-".join(self.config["cell"]["gene_feature"]), "-".join([str(v) for v in self.config["cell"]["featurizer"].values()])))
        if osp.exists(save_path):
            self.cell_dict = pickle.load(open(save_path, "rb"))
        else:
            self.cell_dict = {}
            for feat in self.config["cell"]["gene_feature"]:
                data_cell = np.loadtxt(osp.join(self.path, feat2file[feat] + ".csv"), dtype=str, delimiter=',')
                gene_names = data_cell[0][1:]
                selected_idx = [0] + self.gene_selector(gene_names)
                data_cell = data_cell[1:, selected_idx]
                cur_cell_feat = {}
                for cell in data_cell:
                    cell_name = cell[0][:12]
                    if cell_name not in self.cell_dict:
                        cur_cell_feat[cell_name] = cell[1:].reshape(-1, 1).astype(np.float)
                    else:
                        cur_cell_feat[cell_name] = np.concatenate((cur_cell_feat[cell_name], cell[1:].reshape(-1, 1).astype(np.float)), axis=1)
                for key in cur_cell_feat:
                    value = np.mean(cur_cell_feat[key], axis=1).reshape(-1, 1)
                    if key not in self.cell_dict:
                        self.cell_dict[key] = value
                    else:
                        self.cell_dict[key] = np.concatenate((self.cell_dict[key], value), axis=1)
            pickle.dump(self.cell_dict, open(save_path, "wb"))

        # load drug and its response data
        df = pd.read_csv(osp.join(self.path, "tcga_clinical_data", self.subset + ".csv"))
        drugs = df["smiles"].unique()
        self.drug_dict = dict(zip(drugs, drugs))
        self.drug_index = df["smiles"].to_numpy()
        self.cell_index = df["bcr_patient_barcode"].to_numpy()
        self.response = df["label"].to_numpy().astype(np.float)

class GDSC2(DRPDataset):
    def __init__(self, path, config, task="regression"):
        super(GDSC2, self).__init__(path, config, task)
        self._train_test_split()
        
    def _load_data(self):
        pertid_ach_smiles_ic50s = json.load(open(osp.join(self.path, "gdsc.json")))
        self.drug_index = [i[2] for i in pertid_ach_smiles_ic50s]
        self.drug_dict = {}
        for smiles in set(self.drug_index):
            self.drug_dict[smiles] = smiles
        self.drug_index = np.array(self.drug_index)
        self.cell_index = [i[1] for i in pertid_ach_smiles_ic50s]
        self.cell_index = np.array(self.cell_index)
        self.IC = [float(i[-1]) for i in pertid_ach_smiles_ic50s]
        if 'lnIC' in self.config and self.config['lnIC']:
            self.IC = [np.log(ic) for ic in self.IC]
        self.IC = np.array(self.IC)
        self.response = np.zeros_like(self.IC)
        
        if 'ach2vec' in self.config['cell']:
            self.cell_dict = json.load(open(osp.join(self.path, self.config['cell']['ach2vec'])))
        else:
            self.cell_dict = json.load(open(osp.join(self.path, "ach2gene.json")))
        for k in self.cell_dict:
            self.cell_dict[k] = np.array(self.cell_dict[k])


SUPPORTED_DRP_DATASET = {
    "GDSC": GDSC,
    "TCGA": TCGA,
    "GDSC2": GDSC2
}