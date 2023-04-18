from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import os
import copy
import scanpy
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset

from feat.cell_featurizer import SUPPORTED_CELL_FEATURIZER

class CTCDataset(Dataset, ABC):
    def __init__(self, path, config, seed):
        super(CTCDataset, self).__init__()
        self.config = config
        self.path = path
        self.seed = seed
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        feat_config = self.config["cell"]["featurizer"]["structure"]
        featurizer = SUPPORTED_CELL_FEATURIZER[feat_config["name"]](feat_config)
        self.cells = [featurizer(cell) for cell in self.cells]

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.cells = [self.cells[i] for i in indexes]
        new_dataset.labels = [self.labels[i] for i in indexes]
        return new_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.cells[index], self.labels[index]

class Zheng68k(CTCDataset):
    def __init__(self, path, config, seed):
        super(Zheng68k, self).__init__(path, config, seed)
        self._train_test_split(seed)

    def _load_data(self):
        data = scanpy.read_h5ad(os.path.join(self.path, "Zheng68K.h5ad"))
        self.cells = data.X
        self.label_dict, self.labels = np.unique(np.array(data.obs['celltype']), return_inverse=True)
        self.num_classes = self.label_dict.shape[0]

    def _train_test_split(self, seed):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(self.cells, self.labels)
        for train_index, val_index in split:
            self.train_index = train_index
            self.val_index = val_index

SUPPORTED_CTC_DATASETS = {
    "zheng68k": Zheng68k,
}        