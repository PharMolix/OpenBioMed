from typing import Optional

import pytorch_lightning as pl

from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class ProteinFolding(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage():
        return "\n".join([
            'Protein folding prediction.',
            'Inputs: {"protein": Protein (an OpenBioMed Protein object)}',
            "Outputs: Protein (an OpenBioMed Protein object with 3D structure available)"
        ])

    
    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("protein_folding", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("protein_folding", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return FoldingEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/rmsd",
            output_str="-{val_rmsd:.4f}",
            mode="min",
        )

class FoldingEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    # TODO: Implement the evaluation logic