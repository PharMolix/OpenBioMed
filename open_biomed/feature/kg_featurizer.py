from abc import ABC, abstractmethod

import torch

from open_biomed.feature.base_featurizer import BaseFeaturizer
from open_biomed.utils.kg_utils import SUPPORTED_KG, embed

class KGFeaturizer(BaseFeaturizer, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO：self.kg is no use
        # self.kg = SUPPORTED_KG[self.config["kg_name"]](self.config["kg_path"])
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


class KGIDFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config["embed_dim"]
        # TODO: hard code
        self.max_index = 49111

    # data: SMILES
    def __call__(self, data):
        if self.transform is not None:
            index = self.transform[data]
            if index == -1 or index is None:
                index = self.max_index
            return index
        else:
            return None

# ugly, redesign later
class KGEFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)
        self.kge = config["kge"]
        self.embed_dim = config["embed_dim"]

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        if data is None or data not in self.kge:
            return torch.zeros(self.embed_dim)
        else:
            return torch.FloatTensor(self.kge[data])


SUPPORTED_KG_FEATURIZER = {
    "id": KGIDFeaturizer,
    "KGE": KGEFeaturizer
}