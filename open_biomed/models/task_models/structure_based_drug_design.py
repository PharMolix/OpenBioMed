from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Pocket, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer, Featurized
from open_biomed.utils.misc import sub_dict

class StructureBasedDrugDesignModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["structure_based_drug_design"] = {
            "forward_fn": self.forward_structure_based_drug_design,
            "predict_fn": self.predict_structure_based_drug_design,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["pocket"]),
                "label": self.featurizers["molecule"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["pocket"]),
                "label": self.collators["molecule"]
            })
        }
    
    @abstractmethod
    def forward_structure_based_drug_design(self, 
        pocket: Featurized[Pocket], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_structure_based_drug_design(self,
        pocket: Featurized[Pocket], 
    ) -> List[Molecule]:
        raise NotImplementedError

class StructureTextBasedMoleculeOptimizationModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["structure_text_based_molecule_optimization"] = {
            "forward_fn": self.forward_structure_text_based_molecule_optimization,
            "predict_fn": self.predict_structure_text_based_molecule_optimization,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["molecule", "pocket", "text"]),
                "label": self.featurizers["molecule"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["molecule", "pocket", "text"]),
                "label": self.collators["molecule"]
            })
        }
    
    @abstractmethod
    def forward_structure_text_based_molecule_optimization(self, 
        molecule: Featurized[Molecule], 
        pocket: Featurized[Pocket],
        text: Featurized[Text],
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_structure_text_based_molecule_optimization(self,
        molecule: Featurized[Molecule], 
        pocket: Featurized[Pocket],
        text: Featurized[Text],
    ) -> List[Molecule]:
        raise NotImplementedError