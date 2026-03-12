from typing import Dict, List, Tuple, Optional
from typing_extensions import Any

import json
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class GoGuidedProteinGeneration(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Go-guided protein generation.',
            'Inputs: A list of GO terms. Example: ["GO:0005515", "GO:0005516", "GO:0005517"].',
            "Outputs: A new protein sequence that best fits the GO terms."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("go_guided_protein_generation", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("go_guided_protein_generation", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return GoGuidedProteinGenerationEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        pass
    
class GoGuidedProteinGenerationEvaluationCallback(pl.Callback):
    pass