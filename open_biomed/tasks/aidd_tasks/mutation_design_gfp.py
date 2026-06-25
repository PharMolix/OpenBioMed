"""Mutation Design GFP Task.

Task for designing high-fluorescence GFP mutants through multi-round iterative
optimization. Mirrors the AAV task; the optimization logic lives in the tool.
"""

from typing import Optional, Any

import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.callbacks import TextOverlapEvalCallback
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

logger = logging.getLogger('OpenBioMed')


class MutationDesignGFP(BaseTask):
    """Task for designing high-fluorescence GFP mutants.

    This task performs multi-round iterative optimization to discover
    mutants with improved fluorescence and high sequence diversity.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'GFP Mutation Design - High-fluorescence mutant proposal.',
            'Inputs: {"num_rounds": number of optimization rounds (default: 10),',
            '         "population_size": number of mutants per round (default: 96),',
            '         "max_mutations": max point mutations per sequence (default: 4),',
            '         "diversity_weight": weight for diversity in selection (default: 0.1)}',
            'Outputs: CSV file with 96 GFP mutant sequences sorted by predicted fitness.'
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        # This task doesn't use traditional datasets
        return DefaultDataModule("mutation_design_gfp", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        # This task uses the tool directly, not a trained model
        return DefaultModelWrapper("mutation_design_gfp", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config] = None) -> pl.Callback:
        return MutationDesignGFPEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="best_fitness",
            output_str="-BestFitness_{best_fitness:.4f}",
            mode="max",
        )


class MutationDesignGFPEvaluationCallback(pl.Callback):
    """Callback for evaluating mutation design results."""

    def __init__(self) -> None:
        super().__init__()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        if batch_idx == 0:
            logger.info(f"Mutation design evaluation completed")
