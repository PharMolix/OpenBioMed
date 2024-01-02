from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import copy
import numpy as np
import json
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from open_biomed.feature.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from open_biomed.utils.kg_utils import subgraph_sample

class PPIDataset(Dataset, ABC):
    def __init__(self, path, config, directed=False, make_network=False, split='random'):
        super(PPIDataset, self).__init__()
        self.path = path
        self.config = config
        self.directed = directed
        self.make_network = make_network
        self._load_proteins()
        self._load_ppis()
        self._featurize()
        self._train_test_split(strategy=split)

    @abstractmethod
    def _load_proteins(self, path):
        raise NotImplementedError

    @abstractmethod
    def _load_ppis(self):
        raise NotImplementedError

    def _featurize(self):
        logger.info("Featurizing...")
        if len(self.config["modality"]) > 1:
            featurizer = ProteinMultiModalFeaturizer(self.config["featurizer"])
        else:
            featurizer = SUPPORTED_PROTEIN_FEATURIZER[self.config["featurizer"]["structure"]["name"]](self.config["featurizer"]["structure"])
        self.proteins = [featurizer(protein) for protein in tqdm(self.proteins)]
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    @abstractmethod
    def _train_test_split(self, strategy='random'):
        raise NotImplementedError

    def index_select(self, indexes, split='train'):
        new_dataset = copy.deepcopy(self)
        new_dataset.pair_index = [self.pair_index[i] for i in indexes]
        new_dataset.labels = [self.labels[i] for i in indexes]
        if self.make_network:
            if split == 'train':
                # inductive setting, remove edges in the test set during training
                new_dataset.network = Data(
                    x=torch.stack(self.proteins), 
                    edge_index=torch.tensor(np.array(new_dataset.pair_index).T, dtype=torch.long)
                )
            else:
                new_dataset.network = Data(
                    x=torch.stack(self.proteins),
                    edge_index=torch.tensor(np.array(self.pair_index).T, dtype=torch.long)
                )
        return new_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.make_network:
            return self.proteins[self.pair_index[idx][0]], self.proteins[self.pair_index[idx][1]], self.labels[idx]
        else:
            return self.pair_index[idx][0], self.pair_index[idx][1], self.labels[idx]

class STRINGDataset(PPIDataset):
    def __init__(self, path, config, directed=False, make_network=False, split='bfs'):
        super(STRINGDataset, self).__init__(path, config, directed, make_network, split)
        self.num_classes = 7

    def _load_proteins(self):
        self.proteins = []
        self.protname2id = {}
        with open(osp.join(self.path, "sequences.tsv")) as f:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip("\n").split("\t")
                self.protname2id[line[0]] = i
                self.proteins.append(line[1])
        logger.info("Num proteins: %d" % (len(self.proteins)))

    def _load_ppis(self):
        ppi_dict = {}
        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5, 'expression': 6}
        with open(osp.join(self.path, "interactions.txt")) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip("\t").split("\t")
                prot1, prot2 = self.protname2id[line[0]], self.protname2id[line[1]]
                if not self.directed and prot1 > prot2:
                    t = prot1
                    prot1 = prot2
                    prot2 = t
                if (prot1, prot2) not in ppi_dict:
                    ppi_dict[(prot1, prot2)] = [0] * 7
                ppi_dict[(prot1, prot2)][class_map[line[2]]] = 1
        self.pair_index = []
        self.labels = []
        for prot_pair in ppi_dict:
            self.pair_index.append(list(prot_pair))
            self.labels.append(ppi_dict[prot_pair])
            if not self.directed:
                self.pair_index.append([prot_pair[1], prot_pair[0]])
                self.labels.append(ppi_dict[prot_pair])
        logger.info("Num ppis: %d" % (len(self.labels)))

    def _train_test_split(self, strategy='bfs', test_ratio=0.2, random_new=False):
        if random_new or not osp.exists(osp.join(self.path, "split.json")):
            self.test_indexes = subgraph_sample(len(self.proteins), self.pair_index, strategy, int(len(self.pair_index) * test_ratio), directed=False)
            self.train_indexes = []
            for i in range(len(self.pair_index)):
                if i not in self.test_indexes:
                    self.train_indexes.append(i)
            json.dump({
                "train": self.train_indexes,
                "test": self.test_indexes
            }, open(osp.join(self.path, "split_%s.json" % (strategy)), "w"))
        else:
            split = json.load(open(osp.join(self.path, "split_%s.json" % (strategy)), "r"))
            self.train_indexes = split["train"]
            self.test_indexes = split["test"]

SUPPORTED_PPI_DATASETS = {
    "SHS27k": STRINGDataset,
    "SHS148k": STRINGDataset,
    "STRING": STRINGDataset,
}