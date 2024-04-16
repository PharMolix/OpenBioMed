from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import copy
from enum import Enum
import numpy as np
import pandas as pd
import os
from rdkit import Chem
import os
import sys

import torch
from torch.utils.data import Dataset
from rdkit.Chem import AllChem, Descriptors

from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer
from open_biomed.utils.kg_utils import SUPPORTED_KG, embed
from open_biomed.utils.split_utils import random_split, scaffold_split
from open_biomed.utils import Normalizer

sys.path.insert(0, os.path.dirname(__file__))

def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_malaria_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['activity']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_cep_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['PCE']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def check_columns(df, tasks, N):
    bad_tasks = []
    total_missing_count = 0
    for task in tasks:
        value_list = df[task]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        total_missing_count += missing_count
        pos_ratio = 100. * pos_count / (pos_count + neg_count)
        missing_ratio = 100. * missing_count / N
        assert pos_count + neg_count + missing_count == N
        if missing_ratio >= 50:
            bad_tasks.append(task)
        print('task {}\t\tpos_ratio: {:.5f}\tmissing ratio: {:.5f}'.format(task, pos_ratio, missing_ratio))
    print('total missing ratio: {:.5f}'.format(100. * total_missing_count / len(tasks) / N))
    return bad_tasks


def check_rows(labels, N):
    from collections import defaultdict
    p, n, m = defaultdict(int), defaultdict(int), defaultdict(int)
    bad_count = 0
    for i in range(N):
        value_list = labels[i]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        p[pos_count] += 1
        n[neg_count] += 1
        m[missing_count] += 1
        if pos_count + neg_count == 0:
            bad_count += 1
    print('bad_count\t', bad_count)
    
    print('pos\t', p)
    print('neg\t', n)
    print('missing\t', m)
    return


def _load_pcba_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    tasks = list(input_df.columns)[:-2]

    N = input_df.shape[0]
    temp_df = input_df[tasks]
    temp_df = temp_df.replace(0, -1)
    temp_df = temp_df.fillna(0)

    bad_tasks = check_columns(temp_df, tasks, N)
    for task in bad_tasks:
        tasks.remove(task)
    print('good tasks\t', len(tasks))

    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    labels = labels.values
    print(labels.shape)  # 439863, 92
    check_rows(labels, N)

    input_df.dropna(subset=tasks, how='all', inplace=True)
    # convert 0 to -1
    # input_df = input_df.replace(0, -1)
    # convert nan to 0
    input_df = input_df.fillna(0)
    labels = input_df[tasks].values
    print(input_df.shape)  # 435685, 92
    N = input_df.shape[0]
    check_rows(labels, N)

    smiles_list = input_df['smiles'].tolist()
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_sider_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values

# root_path = 'dataset/chembl_with_labels'
def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively """

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one """

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    # return smiles_list, rdkit_mol_objs_list, folds.values, labels.values
    return smiles_list, rdkit_mol_objs_list, labels.values

datasetname2function = {
    "bbbp": _load_bbbp_dataset,
    "clintox": _load_clintox_dataset,
    "tox21": _load_tox21_dataset,
    "toxcast": _load_toxcast_dataset,
    "sider": _load_sider_dataset,
    "hiv": _load_hiv_dataset,
    "bace": _load_bace_dataset,
    "muv": _load_muv_dataset,
    "freesolv": _load_freesolv_dataset,
    "esol": _load_esol_dataset,
    "lipophilicity": _load_lipophilicity_dataset,
}

class Task(Enum):
    CLASSFICATION = 0
    REGRESSION = 1

class DPDataset(Dataset, ABC):
    def __init__(self, path, config, in_memory=True):
        super(DPDataset, self).__init__()
        self.path = path
        self.config = config
        self.in_memory = in_memory
        self._load_data()
        # self._featurize()

    @abstractmethod
    def _load_data(self, path):
        raise NotImplementedError

    def _featurize(self):
        logger.info("Featurizing...")
        # self.featurized_drugs: 如果是多模态就是一个dict, 如果是structure单模态就是list[Data()]
        self.drugs = [self.drug_featurizer(drug) for drug in self.drugs]
        self.labels = [torch.tensor(label) for label in self.labels]

    def _build(self, save_path=""):
        if len(self.config["mol"]["modality"]) > 1 and save_path:
            kg_config = self.config["mol"]["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            self.drug2kg, self.drug2text, _, _ = self.kg.link(self)
            # TODO: dp use TransE, don't need filter_out?
            filter_out = []
            """
            for i_drug in data_index:
                smi = self.smiles[i_drug]
                #if smi in self.drug2kg:
                #    filter_out.append((self.drug2kg[smi], self.protein2kg[protein]))
            """
            # embed once for consistency
            try:
                kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], save=True, save_path=save_path)
            except Exception as e:
                kge = None
            self.config["mol"]["featurizer"]["kg"]["kge"] = kge
        self._configure_featurizer()
        # featurize all data pairs in one pass for training efficency
        if self.in_memory:
            self._featurize()

    def _configure_featurizer(self):
        if len(self.config["mol"]["modality"]) > 1:
            self.drug_featurizer = MolMultiModalFeaturizer(self.config["mol"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            self.drug_featurizer.set_drug2text_dict(self.drug2text)
        else:
            drug_feat_config = self.config["mol"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_MOL_FEATURIZER[drug_feat_config["name"]](drug_feat_config)

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        new_dataset.labels = [new_dataset.labels[i] for i in indexes]
        return new_dataset

    def __getitem__(self, index):
        if not self.in_memory:
            drug, label = self.drugs[index], self.labels[index]
            return self.drug_featurizer(drug), label
        else:
            return self.featurized_drugs[index], self.labels[index]

    def __len__(self):
        return len(self.drugs)

class MoleculeNetDataset(DPDataset):
    name2target = {
        "BBBP":     ["p_np"],
        "Tox21":    ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
                     "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],
        "ClinTox":  ["CT_TOX", "FDA_APPROVED"],
        "HIV":      ["HIV_active"],
        "Bace":     ["class"],
        "SIDER":    ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
                     "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
                     "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
                     "Reproductive system and breast disorders", 
                     "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
                     "General disorders and administration site conditions", "Endocrine disorders", 
                     "Surgical and medical procedures", "Vascular disorders", 
                     "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
                     "Congenital, familial and genetic disorders", "Infections and infestations", 
                     "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
                     "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
                     "Ear and labyrinth disorders", "Cardiac disorders", 
                     "Nervous system disorders", "Injury, poisoning and procedural complications"],
        "MUV":      ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
                     'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
                     'MUV-652', 'MUV-466', 'MUV-832'],
        "Toxcast":  [""], # 617
        "FreeSolv": ["expt"],
        "ESOL":     ["measured log solubility in mols per litre"],
        "Lipo":     ["exp"],
        "qm7":      ["u0_atom"],
        "qm8":      ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
                     "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"],
        "qm9":      ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
    }
    name2task = {
        "BBBP":     Task.CLASSFICATION,
        "Tox21":    Task.CLASSFICATION,
        "ClinTox":  Task.CLASSFICATION,
        "HIV":      Task.CLASSFICATION,
        "Bace":     Task.CLASSFICATION,
        "SIDER":    Task.CLASSFICATION,
        "MUV":      Task.CLASSFICATION,
        "Toxcast":  Task.CLASSFICATION,
        "FreeSolv": Task.REGRESSION,
        "ESOL":     Task.REGRESSION,
        "Lipo":     Task.REGRESSION,
        "qm7":      Task.REGRESSION,
        "qm8":      Task.REGRESSION,
        "qm9":      Task.REGRESSION
    }

    def __init__(self, path, config, name="BBBP", label_type=1):
        if name not in self.name2target:
            raise ValueError("%s is not a valid moleculenet task!" % name)
        file_name = os.listdir(os.path.join(path, name.lower(), "raw"))[0]
        assert file_name[-4:] == ".csv"
        path = os.path.join(path, name.lower(), "raw", file_name)
        self.name = name
        self.targets = self.name2target[name]
        # TODO: del: no use
        self.task = self.name2task[name]
        # TODO: del label_type
        self.label_type = label_type
        super(MoleculeNetDataset, self).__init__(path, config)
        self._train_test_split()
        self._normalize()
        
    
    def _load_data(self):
        smiles_list, rdkit_mol_objs, labels = datasetname2function[self.name.lower()](self.path)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        self.smiles, self.drugs, self.labels = [], [], []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is None:
                continue
            # TODO: drugs and smiles are all get from AllChem.MolFromSmiles()
            self.smiles.append(smiles_list[i])
            self.drugs.append(smiles_list[i])
            # self.drugs.append(rdkit_mol[i])
            self.labels.append(labels[i])
    
    def _train_test_split(self, strategy="scaffold"):
        if strategy == "random":
            self.train_index, self.val_index, self.test_index = random_split(len(self), 0.1, 0.1)
        elif strategy == "scaffold":
            self.train_index, self.val_index, self.test_index = scaffold_split(self, 0.1, 0.1, is_standard=True)

    def _normalize(self):
        if self.name in ["qm7", "qm9"]:
            self.normalizer = []
            for i in range(len(self.targets)):
                self.normalizer.append(Normalizer(self.labels[:, i]))
                self.labels[:, i] = self.normalizer[i].norm(self.labels[:, i])
        else:
            # TODO:
            self.normalizer = [None] * len(self.targets)

SUPPORTED_DP_DATASETS = {
    "MoleculeNet": MoleculeNetDataset
}