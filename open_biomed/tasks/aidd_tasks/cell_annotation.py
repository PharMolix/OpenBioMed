from typing import Dict, List, Tuple, Optional
from typing_extensions import Any

import json
import logging
import os
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from open_biomed.data import molecule_fingerprint_similarity, check_identical_molecules
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class CellAnnotation(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage():
        return "\n".join([
            'Cell annotation.',
            'Inputs: {"cell": Cell (an OpenBioMed Cell object), "class_texts": List[Text] (text descriptions of the cell types)}',
            "Outputs: int (the index of the predicted cell type)"
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("cell_annotation", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("cell_annotation", model_cfg, train_cfg)

    # @staticmethod
    # def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
    #     return MoleculePropertyPredictionEvaluationCallback()

    # @staticmethod
    # def get_monitor_cfg() -> Struct:
    #     return Struct(
    #         name="val/roc_auc",
    #         output_str="-{val_roc_auc:.4f}",
    #         mode="max",
    #     )
