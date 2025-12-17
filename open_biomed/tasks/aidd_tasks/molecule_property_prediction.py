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
from scipy.stats import spearmanr


from open_biomed.data import molecule_fingerprint_similarity, check_identical_molecules
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class MoleculePropertyPrediction(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage():
        return "\n".join([
            'Molecular property prediction.',
            'Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}',
            "Outputs: float (the property score of the molecule)"
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("molecule_property_prediction", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("molecule_property_prediction", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return MoleculePropertyPredictionEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/roc_auc",
            output_str="-{val_roc_auc:.4f}",
            mode="max",
        )


class MoleculePropertyPredictionEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None
        self.eval_metric = roc_auc_score

    def on_validation_batch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[STEP_OUTPUT], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)
        if batch_idx == 0:
            for i in range(2):
                out_labels = '\n'.join([str(x) for x in self.eval_dataset.labels[i]])
                logging.info(f"Original: {self.eval_dataset.molecules[i]}")
                logging.info(f"Predict: {self.outputs[i]}")
                logging.info(f"Ground Truth: {out_labels}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:

        y_true = np.squeeze(np.array(self.eval_dataset.labels), axis=1)

        self.outputs = [torch.tensor(x) for x in self.outputs]
        
        # y_scores =  torch.unsqueeze(torch.tensor(self.outputs), dim=1).numpy()
        y_scores_tensor = torch.stack(self.outputs, dim=0)
        y_scores =  y_scores_tensor.cpu().numpy()

        output_str = "\t".join(["roc_auc"]) + "\n"

        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(self.eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            logging.info(len(roc_list))
            logging.info('Some target is missing!')
            logging.info('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

        roc_auc = sum(roc_list) / len(roc_list)

        logging.info(f"{self.status}/roc_auc: {roc_auc:.4f}")

        output_str += "\t".join([f"{roc_auc:.4f}"]) + "\n"

        # 更新监控值
        pl_module.log_dict({f"{self.status}/roc_auc": roc_auc})

        output_path = os.path.join(trainer.default_root_dir, f"{self.status}_outputs", f"epoch{pl_module.current_epoch}")

        # 保存测试结果
        if trainer.is_global_zero:
            with open(output_path + "_outputs.txt", "w") as f:
                f.write(output_str)
        
    def on_test_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.status = "test"
        self.outputs = []
        self.eval_dataset = trainer.test_dataloaders.dataset

    def on_test_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)


class MoleculePropertyPredictionRegression(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod    
    def print_usage():
        return "\n".join([
            'Molecular property prediction.',
            'Inputs: {"molecule": a small molecule}',
            "Outputs: A float number in indicating the regression result of the molecule's certain properties."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("molecule_property_prediction", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("molecule_property_prediction", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        # TODO：传个参数切换 回归与分类任务的评估方法
        return MoleculePropertyPredictionRegressionEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/mae",
            output_str="-{mae:.4f}",
            mode="min",
        )
        # return Struct(
        #     name="val/spearman",
        #     output_str="-{val_spearman:.4f}",
        #     mode="max",
        # )

class MoleculePropertyPredictionRegressionEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None
        # self.eval_metric = roc_auc_score

    def on_validation_batch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[STEP_OUTPUT], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)
        if batch_idx == 0:
            for i in range(2):
                out_labels = '\n'.join([str(x) for x in self.eval_dataset.labels[i]])
                logging.info(f"Original: {self.eval_dataset.molecules[i]}")
                logging.info(f"Predict: {self.outputs[i]}")
                logging.info(f"Ground Truth: {out_labels}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:

        y_true = np.squeeze(np.array(self.eval_dataset.labels), axis=1)
        
        # y_scores =  torch.unsqueeze(torch.tensor(self.outputs), dim=1).numpy()
        # print(self.outputs)
        # print(y_true)
        # print(y_true.shape)
        # print(len(self.outputs))

        tensors = [torch.tensor(x) for x in self.outputs]
        y_pred_tensor = torch.stack(tensors, dim=0)
        y_pred = y_pred_tensor.cpu().numpy()

        # Prepare metrics for regression
        output_str = "\t".join(["rmse", "mae", "r2", "spearman"]) + "\n"

        rmse_list = []
        mae_list = []
        r2_list = []
        spearman_list = []

        for i in range(y_true.shape[1]):
            is_valid = ~np.isnan(y_true[:, i])  # Check for valid targets
            y_true_col = y_true[is_valid, i]
            y_pred_col = y_pred[is_valid, i]
            
            if len(y_true_col) > 0:  # Only calculate if we have valid samples
                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_true_col - y_pred_col) ** 2))
                rmse_list.append(rmse)
                
                # Calculate MAE
                mae = np.mean(np.abs(y_true_col - y_pred_col))
                mae_list.append(mae)
                
                # Calculate R²
                ss_res = np.sum((y_true_col - y_pred_col) ** 2)
                ss_tot = np.sum((y_true_col - np.mean(y_true_col)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Small epsilon to avoid division by zero
                r2_list.append(r2)
                
                # Calculate Spearman correlation
                spearman_corr, _ = spearmanr(y_true_col, y_pred_col)
                spearman_list.append(spearman_corr)
            else:
                print(f'Column {i} has no valid targets')

        if len(rmse_list) < y_true.shape[1]:
            logging.info(f'Valid targets count: {len(rmse_list)}/{y_true.shape[1]}')
            logging.info('Some target is missing!')
            logging.info('Missing ratio: %f' % (1 - float(len(rmse_list)) / y_true.shape[1]))

        # Calculate average metrics
        avg_rmse = np.mean(rmse_list) if rmse_list else np.nan
        avg_mae = np.mean(mae_list) if mae_list else np.nan
        avg_r2 = np.mean(r2_list) if r2_list else np.nan
        avg_spearman = np.mean(spearman_list) if spearman_list else np.nan

        logging.info(f"{self.status}/rmse: {avg_rmse:.4f}")
        logging.info(f"{self.status}/mae: {avg_mae:.4f}")
        logging.info(f"{self.status}/r2: {avg_r2:.4f}")
        logging.info(f"{self.status}/spearman: {avg_spearman:.4f}")

        output_str += "\t".join([f"{avg_rmse:.4f}", f"{avg_mae:.4f}", f"{avg_r2:.4f}", f"{avg_spearman:.4f}"]) + "\n"

        # Log metrics
        pl_module.log_dict({
            f"{self.status}/rmse": avg_rmse,
            f"{self.status}/mae": avg_mae,
            f"{self.status}/r2": avg_r2,
            f"{self.status}/spearman": avg_spearman
        })

        # Save results
        output_path = os.path.join(trainer.default_root_dir, f"{self.status}_outputs", f"epoch{pl_module.current_epoch}")
        if trainer.is_global_zero:
            with open(output_path + "_outputs.txt", "w") as f:
                f.write(output_str)


        
    def on_test_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.status = "test"
        self.outputs = []
        self.eval_dataset = trainer.test_dataloaders.dataset

    def on_test_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)
