from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from open_biomed.data.protein import Protein
import torch

from open_biomed.models.base_model import BaseModel
from open_biomed.utils.config import Config


class GoGuidedProteinGenerationModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["go_guided_protein_generation"] = {
            "forward_fn": self.forward_go_guided_protein_generation,
            "predict_fn": self.predict_go_guided_protein_generation,
        }
    
    @abstractmethod
    def forward_go_guided_protein_generation(self,
        go_terms: List[List[str]],
        label: List[Protein],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def predict_go_guided_protein_generation(self,
        go_terms: List[str], 
    ) -> Protein:
        raise NotImplementedError