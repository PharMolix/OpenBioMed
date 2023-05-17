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
import pickle

import torch
from torch.utils.data import Dataset
from rdkit.Chem import AllChem, Descriptors

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from utils.kg_utils import SUPPORTED_KG, embed
from utils.split import random_split, scaffold_split
from utils.utils import Normalizer

sys.path.insert(0, os.path.dirname(__file__))
import dp_dataset_load as dpload


class Task(Enum):
    CLASSFICATION = 0
    REGRESSION = 1

class DPDataset(Dataset, ABC):
    def __init__(self, path, config, in_memory=True):
        super(DPDataset, self).__init__()
        self.path = path
        self.config = config
        self.in_memory = in_memory
        # self._featurize()

    @abstractmethod
    def _load_data(self, path):
        raise NotImplementedError

    def _featurize(self, save=True):
        logger.info("Featurizing...")
        # self.featurized_drugs: 如果是多模态就是一个dict, 如果是structure单模态就是list[Data()]
        self.featurized_drugs = [self.drug_featurizer(drug) for drug in self.drugs]
        self.labels = [torch.tensor(label) for label in self.labels]
        if save:
            torch.save((self.featurized_drugs, self.labels, self.drugs, self.train_index, self.val_index, self.test_index), self.processed_file_path)

    def _load_kg(self):
        
        if len(self.config["drug"]["modality"]) > 1 and "kg" in self.config["drug"]["modality"]:
            kg_config = self.config["drug"]["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            # TODO: 
            if kg_config["kg_name"] == "BMKG":
                self.drug2kg, self.drug2text, _, _ = self.kg.link(self)
            else:
                self.drug2kg, self.drug2text, _, _ = self.kg.link_chebi20(self)
            # TODO: dp use TransE, don't need filter_out?
            filter_out = []
            """
            for i_drug in data_index:
                smi = self.smiles[i_drug]
                #if smi in self.drug2kg:
                #    filter_out.append((self.drug2kg[smi], self.protein2kg[protein]))
            """
            # embed once for consistency
            if kg_config["kg_name"] == "BMKG":
                try:
                    save_path = self.config["drug"]["featurizer"]["kg"]["save_path"]
                    kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], save=True, save_path=save_path)
                except Exception as e:
                    kge = None
            else:
                kge = None
            self.config["drug"]["featurizer"]["kg"]["kge"] = kge

    def _build(self):
        # featurize all data pairs in one pass for training efficency
        # TODO: 如果已经有了现成的文件直接加载一下就好了，没有的话还有load_kg
        self._load_kg()
        self._configure_featurizer()
        if self.in_memory:
            self._featurize()

    def _configure_featurizer(self, save_path=""):
        
        if len(self.config["drug"]["modality"]) > 1 and "kg" in self.config["drug"]["modality"]:
            self.drug_featurizer = DrugMultiModalFeaturizer(self.config["drug"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            self.drug_featurizer.set_drug2text_dict(self.drug2text)
        else:
            drug_feat_config = self.config["drug"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_DRUG_FEATURIZER[drug_feat_config["name"]](drug_feat_config)

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        new_dataset.labels = [new_dataset.labels[i] for i in indexes]
        new_dataset.featurized_drugs = [new_dataset.featurized_drugs[i] for i in indexes]
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

    def __init__(self, fold_path, config, name="BBBP"):
        if name not in self.name2target:
            raise ValueError("%s is not a valid moleculenet task!" % name)
        file_name = os.listdir(os.path.join(fold_path, name.lower(), "raw"))[0]
        assert file_name[-4:] == ".csv"
        path = os.path.join(fold_path, name.lower(), "raw", file_name)
        self.name = name
        self.targets = self.name2target[name]
        # TODO: del: no use
        self.task = self.name2task[name]
        super(MoleculeNetDataset, self).__init__(path, config)
        self.processed_file_path = self.get_processed_file_path(fold_path, name.lower(), config)
        if os.path.exists(self.processed_file_path):
            self.featurized_drugs, self.labels, self.drugs, self.train_index, self.val_index, self.test_index = torch.load(self.processed_file_path)
        else:
            self._load_data()
            self._train_test_split()
            self._normalize()
            self._build()
        
        #debug
        """    
        index_dict = {"train": self.train_index, "val": self.val_index, "test": self.test_index}
        with open("index.pkl", "wb") as f:
            pickle.dump(index_dict, f)
        """
     
    @property
    def processed_kg_file_names(self):
        return 'kg_processed.pkl'
      
    def get_processed_file_path(self, fold_path, name, config):
        """
        get the .pt file path based on modality
        Args:
            fold_path (str): flod path of MoleculeNet
            
            config (dict): the config of task which used to get the modality
        """
        modality = config["drug"]["modality"]
        assert "structure" in modality
        if len(modality) == 2 and "kg" in modality:
            return os.path.join(fold_path, name.lower(), "processed", "data_processed_kg.pt")
        elif len(modality) == 2 and "text" in modality:
            return os.path.join(fold_path, name.lower(), "processed", "data_processed_text.pt")
        elif len(modality) == 3 and config["drug"]["featurizer"]["text"]["transformer_type"] == "gpt2":
            return os.path.join(fold_path, name.lower(), "processed", "data_processed_biomedgpt.pt")
        elif len(modality) == 3 and config["drug"]["featurizer"]["kg"]["kg_name"] == "BMKG":
            return os.path.join(fold_path, name.lower(), "processed", "data_processed_bmkgv1.pt")
        elif len(modality) == 3:
            return os.path.join(fold_path, name.lower(), "processed", "data_processed_text_kg.pt")
        elif len(modality) == 1 and config["drug"]["featurizer"]["structure"]["name"] in ["transformer"]:
            return os.path.join(fold_path, name.lower(), "processed", "data_processed_kvplm.pt")
        else:
            return os.path.join(fold_path, name.lower(), "processed", "data_processed.pt")
    
    def _load_data(self):
        
        smiles_list, rdkit_mol_objs, labels = getattr(globals()["dpload"], f"_load_{self.name.lower()}_dataset")(self.path)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        self.smiles, self.drugs, self.labels, self.mols = [], [], [], []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is None:
                continue
            # TODO: drugs and smiles are all get from AllChem.MolFromSmiles()
            self.smiles.append(smiles_list[i])
            self.drugs.append(smiles_list[i])
            self.mols.append(rdkit_mol_objs[i])
            self.labels.append(labels[i])
    """        
    def _load_data(self):
        # process the raw data which readed from file 
        # return: self.smiles, self.drugs, self.labels
        data = pd.read_csv(self.path)
        smiles = data['smiles'].to_numpy()
        if self.label_type == 1:
            labels = data[self.targets]
            labels = labels.fillna(0)
            labels = labels.to_numpy()
        else:
            labels = data[self.targets]
            # convert 0 to -1
            labels = labels.replace(0, -1)
            # convert nan to 0
            labels = labels.fillna(0)
            labels = labels.to_numpy()
        
        self.smiles, self.drugs, self.labels = [], [], []
        for i, drug in enumerate(smiles):
            mol = AllChem.MolFromSmiles(drug)
            # mol = Chem.MolFromSmiles(drug)

            if mol is not None:
                # self.smiles.append(drug)
                self.smiles.append(AllChem.MolToSmiles(mol))
                self.drugs.append(drug)
                self.labels.append(labels[i])
        # Debug
        
        # data_smiles_series = pd.Series(self.smiles)
        # floder_path = os.path.dirname(self.path)
        # file_name = os.path.join(floder_path, "smiles_2.csv")
        # data_smiles_series.to_csv(file_name, index=False, header=False)
        # TODO:
        if self.name == "qm9":
            for target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
                self.labels[:, self.targets.index(target)] *= 27.211386246
    """    
    
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