from abc import ABC
import logging
logger = logging.getLogger(__name__)

import numpy as np
import os.path as osp

import torch
from torch_geometric.data import Data, Dataset

from feat import *

class DTIDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(DPGraphDataset, self).__init__()
        self.path = path
        self.config = config

    @abstractmethod
    def load_data(self, path):
        raise NotImplementedError

    def featurize(self):
        pass

    def split(splitter='random'):
        pass

    def __len__(self):
        pass

class SHS_Sample(DTIDataset):

class SHS_Full(DTIDataset):
