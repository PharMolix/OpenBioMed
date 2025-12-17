from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
from typing_extensions import Any

import argparse
# from absl import logging
import datetime
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytz
import torch
from tqdm import tqdm

from open_biomed.tools.base_tool import Tool
from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.tasks import TASK_REGISTRY
from open_biomed.utils.callbacks import RecoverCallback, GradientClip
from open_biomed.utils.config import Config, Struct, merge_config
from open_biomed.utils.distributed import setup_outputs_for_distributed

logging_level = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}

class Pipeline(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def setup_infra(self) -> None:
        # Configure basic utilities such as logging and cerating output directories
        raise NotImplementedError

class TrainValPipeline(Pipeline):
    def __init__(self) -> None:
        super(TrainValPipeline, self).__init__()

        # Parse arguments
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)
        args = parser.parse_args()
        # Combine arguments with additional configuration files
        self.cfg = Config(config_file="./configs/basic_config.yaml", **args.__dict__)
        for file in args.additional_config_file:
            merge_config(
                self.cfg,
                Config(file, **args.__dict__)
            )

        # Prepare task
        if self.cfg.task not in TASK_REGISTRY:
            raise NotImplementedError(f"{self.cfg.task} has not been implemented! Current tasks are {[task for task in TASK_REGISTRY.keys()]}")
        self.task = TASK_REGISTRY[self.cfg.task]
        self.cfg.train.monitor = self.task.get_monitor_cfg()

        self.setup_infra()

        # Prepare model
        # NOTE: the model should be prepared in advance for featurizers and collators
        self.setup_model()

        # Prepare dataloaders
        self.setup_data()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Add training arguments required for training
        parser.add_argument("--task", type=str)                                 # Task name
        parser.add_argument("--additional_config_file", type=str, nargs='+')    # Additional configuration files
        parser.add_argument("--exp_name", type=str, default="train")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--wandb", action='store_true')
        parser.add_argument("--wandb_resume_id", type=str, default=None)
        parser.add_argument("--empty_folder", action='store_true')
        parser.add_argument("--test_only", action="store_true")    # Testing only
        parser.add_argument("--num_gpus", type=int, default=1)     # Number of GPUs

        # Global config
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--logging_level", type=str, default="info")
        
        # Dataset config
        parser.add_argument("--dataset_name", type=str)
        parser.add_argument("--dataset_path", type=str)
        parser.add_argument("--num_workers", type=int, default=1)

        # Train & Val params
        parser.add_argument("--batch_size_train", type=int, default=8)
        parser.add_argument("--batch_size_eval", type=int, default=16)
        parser.add_argument("--max_epochs", type=int, default=50)
        parser.add_argument("--ckpt_freq", type=int, default=5)
        parser.add_argument("--save_top_k", type=int, default=5)
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--scheduler", type=str, default="cosine", choices=['cosine', 'plateau'])
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--max_grad_norm", type=str, default='Q')
        parser.add_argument("--log_interval", type=int, default=100)
        parser.add_argument("--ckpt_path", type=str, default="last")

    def setup_infra(self) -> None:
        # Setup paths
        project_name = "-".join([self.cfg.task, self.cfg.model.name, self.cfg.dataset.name])
        accounting_dir = os.path.join("./logs", self.cfg.task, f"{self.cfg.model.name}-{self.cfg.dataset.name}", self.cfg.exp_name)
        if self.cfg.empty_folder:
            os.system(f"rm -r {accounting_dir}")
        if not os.path.exists(accounting_dir):
            os.makedirs(accounting_dir, exist_ok=True)
        dump_config_path = os.path.join(accounting_dir, "config.yaml")
        checkpoint_dir = os.path.join(accounting_dir, "checkpoints")
        val_output_dir = os.path.join(accounting_dir, "val_outputs")
        test_output_dir = os.path.join(accounting_dir, "test_outputs")
        if not os.path.exists(val_output_dir):
            os.makedirs(val_output_dir, exist_ok=True)
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir, exist_ok=True)
        accounting_cfg = {
            "dir": accounting_dir,
            "project_name": project_name,
            "dump_config_path": dump_config_path,
            "checkpoint_dir": checkpoint_dir,
            "val_output_dir": val_output_dir,
            "test_output_dir": test_output_dir,
        }
        self.cfg.accounting = Struct(**accounting_cfg)
        if not self.cfg.test_only:
            self.cfg.save2yaml(os.path.join(accounting_dir, "config.yaml"))

        # Setup seed
        pl.seed_everything(self.cfg.seed, workers=True)

        # Setup parallel training
        self.cfg.train.distributed = self.cfg.train.num_gpus > 1
        if self.cfg.train.distributed:
            local_rank = int(os.environ["RANK"])
            setup_outputs_for_distributed(local_rank == 0, self.cfg)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt=f'{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
            level=logging_level[self.cfg.logging_level],
        )
        if self.cfg.wandb_resume_id is not None:
            self.wandb_logger = WandbLogger(
                id=self.cfg.wandb_resume_id,
                project=self.project_name,
                offline=not self.cfg.wandb,
                save_dir=self.cfg.accounting.dir,
                resume='must',
            )
        else: 
            # Start a new run
            self.wandb_logger = WandbLogger(
                name=f"{self.cfg.task}"
                + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
                offline=not self.cfg.wandb,
                save_dir=self.cfg.accounting.dir,
            )
        self.wandb_logger.log_hyperparams(self.cfg.todict())
        logging.info(f"The config of this process is:\n{self.cfg}")
    
    def setup_data(self) -> None:
        logging.info("Preparing data...")
        self.datamodule = self.task.get_datamodule(
            self.cfg.dataset,
            *self.model.get_featurizer(),
        )
        logging.info("Data preparation finished!")

    def setup_model(self) -> None:
        self.model = self.task.get_model_wrapper(self.cfg.model, self.cfg.train)
        try:
            max_grad_norm = float(self.cfg.train.max_grad_norm)
        except ValueError:
            max_grad_norm = self.cfg.train.max_grad_norm
        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.cfg.train.monitor.name,
            every_n_epochs=self.cfg.train.ckpt_freq,
            dirpath=self.cfg.accounting.checkpoint_dir,
            filename="epoch{epoch:02d}" + self.cfg.train.monitor.output_str,
            save_top_k=self.cfg.train.save_top_k,
            mode=self.cfg.train.monitor.mode,
            auto_insert_metric_name=False,
            save_last=True,
        )
        self.trainer = pl.Trainer(
            default_root_dir=self.cfg.accounting.dir,
            max_epochs=self.cfg.train.max_epochs,
            check_val_every_n_epoch=self.cfg.train.ckpt_freq,
            devices=self.cfg.train.num_gpus,
            strategy="ddp_find_unused_parameters_true" if self.cfg.train.distributed > 1 else "auto",
            logger=self.wandb_logger,
            inference_mode=not self.cfg.test_only,
            num_sanity_val_steps=0,
            log_every_n_steps=self.cfg.train.log_interval,
            callbacks=[
                RecoverCallback(
                    latest_ckpt=os.path.join(self.cfg.accounting.checkpoint_dir, "last.ckpt"),
                    resume=self.cfg.train.resume,
                    recover_trigger_loss=1e7,
                ),
                GradientClip(max_grad_norm=max_grad_norm),
                self.checkpoint_callback,
            ] + [self.task.get_callbacks()]
        )

    def run(self) -> None:
        num_test_repeats = getattr(self.cfg.evaluation, "num_test_repeats", 1)
        if not self.cfg.test_only:
            self.model.train_cfg.max_iters = len(self.datamodule.train_dataloader()) * self.cfg.train.max_epochs
            self.trainer.fit(self.model, train_dataloaders=self.datamodule.train_dataloader(), val_dataloaders=self.datamodule.val_dataloader())
            # ckpt_path can be 'best', 'last', or a specific path
            if self.cfg.evaluation.ckpt_path == "best":
                ckpt_path = self.checkpoint_callback.best_model_path
            elif self.cfg.evaluation.ckpt_path == "last":
                ckpt_path = os.path.join(self.cfg.accounting.checkpoint_dir, "last.ckpt")
            else:
                ckpt_path = self.cfg.evaluation.ckpt_path
        else:
            ckpt_path = self.cfg.evaluation.ckpt_path

        # TODO: implement parallel testing on multiple gpus
        if num_test_repeats > 1:
            # NOTE: for multiple repeats, the results should be averaged over all repeats in ValidationCallback
            for i in range(num_test_repeats):
                logging.info(f"Testing the {i}-th repeat...")
                cur_result = self.trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path)[0]
            print("Overall results:", cur_result)
        else:
            self.trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path)

class InferencePipeline(Pipeline, Tool):
    def __init__(self, 
        task: str="",
        model: str="",
        model_ckpt: str="",
        additional_config: Optional[str]=None,
        logging_level: str="info",
        device: str="cpu",
        output_prompt: Optional[str]=None,
        retry_limit: int=10,
    ) -> None:
        super().__init__()

        self.cfg = Config(config_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "model", f"{model}.yaml"))
        if additional_config is not None:
            self.cfg = merge_config(self.cfg, Config(config_file=additional_config))
        self.cfg.task = task
        self.cfg.model_ckpt = model_ckpt
        self.cfg.device = device
        self.cfg.logging_level = logging_level
        self.best_batch_size = 1
        self.best_batch_size_defined = False
        self.output_prompt = output_prompt
        self.retry_limit = retry_limit

        # Prepare task
        if self.cfg.task not in TASK_REGISTRY:
            raise NotImplementedError(f"{self.cfg.task} has not been implemented! Current tasks are {[task for task in TASK_REGISTRY]}")
        self.task = TASK_REGISTRY[self.cfg.task]

        self.infra_ready = False
        self.model_ready = False

    def setup_infra(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging_level[self.cfg.logging_level],
        )
        logging.info(f"The config of this process is:\n{self.cfg}")

    def setup_model(self):
        self.model = self.task.get_model_wrapper(self.cfg.model, None)
        self.featurizer, self.collator = self.model.get_featurizer()
        if os.path.exists(self.cfg.model_ckpt):
            logging.info(f"Loading model from {self.cfg.model_ckpt}")
            state_dict = torch.load(open(self.cfg.model_ckpt, "rb"), map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if hasattr(self.model.model, "load_ckpt"):
                self.model.model.load_ckpt(state_dict)
            else:
                self.model.load_state_dict(state_dict, strict=False)
        self.model.model.eval()
        self.model.to(self.cfg.device)

    def print_usage(self):
        return self.task.print_usage()

    def run(self, batch_size: Union[int, str]="max", *args, **kwargs) -> Any:
        if not self.infra_ready:
            self.setup_infra()
            self.infra_ready = True
        if not self.model_ready:
            self.setup_model()
            self.model_ready = True
        logging.debug(f"Input: {kwargs}")
        for key in kwargs:
            if not isinstance(kwargs[key], list):
                kwargs[key] = [kwargs[key]]
            num_samples = len(kwargs[key])
        
        # Featurization
        featurized = []
        for i in range(num_samples):
            inputs = {k:v[i] for k, v in kwargs.items()}
            outputs = self.featurizer(**inputs)
            featurized.append(outputs)
            
        if batch_size == "auto":
            # Automatically define batch size
            if not self.best_batch_size_defined and num_samples > self.best_batch_size:
                self.best_batch_size = num_samples
                while True:
                    try:
                        inputs = featurized[:self.best_batch_size]
                        self.model.predict(self.collator(inputs))
                        break
                    except Exception as e:
                        if isinstance(e, RuntimeError) and "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            logging.debug(f"Reduce batch size from {self.best_batch_size} to {self.best_batch_size // 2}")
                            self.best_batch_size = self.best_batch_size // 2
                            self.best_batch_size_defined = True
                        else:
                            raise e
            batch_size = self.best_batch_size
        elif batch_size == "max":
            batch_size = num_samples
        
        outputs = [None for i in range(num_samples)]
        for i in tqdm(range((num_samples - 1) // batch_size + 1), desc="Inference Steps"):
            # Generate a output text that both LLM and experts can understand
            retry_times = 0
            retry_idx = range(i * batch_size, (i + 1) * batch_size)
            while retry_times < self.retry_limit and len(retry_idx) > 0:
                inputs = [featurized[idx] for idx in retry_idx]
                batch_output = self.model.predict(self.model.transfer_batch_to_device(self.collator(inputs), self.cfg.device, 0))
                new_retry_idx = []
                for j, output in enumerate(batch_output):
                    if output is None:
                        # Generation failure, restart
                        new_retry_idx.append(retry_idx[j])
                    else:
                        outputs[retry_idx[j]] = output
                retry_idx = new_retry_idx
                retry_times += 1

        # Save molecules and proteins
        files = []
        for i, output in enumerate(outputs):
            timestamp = int(datetime.datetime.now().timestamp() * 1000)
            file = ""
            if isinstance(output, Molecule):
                file = f"./tmp/mol_{timestamp}_{i}.pkl"
                output.save_binary(file)
            elif isinstance(output, Protein):
                file = f"./tmp/protein_{timestamp}_{i}.pkl"
                output.save_binary(file)
            elif isinstance(output, Pocket):
                file = f"./tmp/pocket_{timestamp}_{i}.pkl"
                output.save_binary(file)
            else:
                file = str(output)
            files.append(file)
        if self.output_prompt:
            return [self.output_prompt.format(output=output, **kwargs) for output in outputs], files
        else:
            return outputs, files

class EnsemblePipeline(Tool):
    def __init__(self, pipelines: Dict[str, InferencePipeline]) -> None:
        super().__init__()
        self.pipelines = pipelines

    def print_usage(self):
        for key, value in self.pipelines.items():
            return value.print_usage()
        
    def run(self, **kwargs):
        task = kwargs.pop("task")
        return self.pipelines[task].run(**kwargs)