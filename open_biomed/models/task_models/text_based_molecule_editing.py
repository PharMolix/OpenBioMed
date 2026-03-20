from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator, ListCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer, IdenticalFeaturizer, Featurized
from open_biomed.utils.misc import sub_dict

class TextBasedMoleculeEditingModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["text_based_molecule_editing"] = {
            "forward_fn": self.forward_text_based_molecule_editing,
            "predict_fn": self.predict_text_based_molecule_editing,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["molecule", "text"]),
                "label": self.featurizers["molecule"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["molecule", "text"]),
                "label": self.collators["molecule"]
            })
        }
    
    @abstractmethod
    def forward_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_text_based_molecule_editing(self,
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        raise NotImplementedError