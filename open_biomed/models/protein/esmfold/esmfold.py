import contextlib
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download
import logging
import os
import torch
import torch.nn as nn
from transformers import EsmTokenizer, DataCollatorWithPadding

from open_biomed.models.protein.esmfold.modeling_esmfold import EsmForProteinFolding
from open_biomed.data import Protein
from open_biomed.models.task_models.protein_folding import ProteinFoldingModel
from open_biomed.utils.collator import Collator, EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class ProteinEsmFeaturizer(Featurizer):
    def __init__(self, 
        protein_tokenizer: EsmTokenizer,
        max_length_protein: int=1024,
        add_special_tokens: bool=False,
    ) -> None:
        super().__init__()
        self.protein_tokenizer = protein_tokenizer
        self.max_length_protein = max_length_protein
        self.add_special_tokens = add_special_tokens

    def __call__(self, protein: Protein) -> Dict[str, Any]:
        sequence = protein.sequence

        featurized = {}
        featurized["protein"] = self.protein_tokenizer(
            sequence,
            max_length=self.max_length_protein,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        )
        
        return featurized
    
    def get_attrs(self) -> List[str]:
        return ["protein"]


class EsmFold(ProteinFoldingModel):
    def __init__(self, model_cfg: Config) -> None:
        super(EsmFold, self).__init__(model_cfg)
        if not os.path.exists(model_cfg.hf_model_name_or_path):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="facebook/esmfold_v1", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
        # load tokenizer
        logging.info("*** loading protein esm tokenizer...")
        self.protein_tokenizer = EsmTokenizer.from_pretrained(model_cfg.hf_model_name_or_path)
        # load esmfold model
        logging.info("*** loading protein folding model...")
        self.protein_model = EsmForProteinFolding.from_pretrained(model_cfg.hf_model_name_or_path, low_cpu_mem_usage=True) 

        if model_cfg.chunk_size:
            self.protein_model.trunk.set_chunk_size(model_cfg.chunk_size)

        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

        self.temp_plddt_res = []

    def maybe_autocast(self, device="cuda:0", dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def featurizer_protein_folding(self) -> Tuple[Featurizer, Collator]:
        return ProteinEsmFeaturizer(
            protein_tokenizer=self.protein_tokenizer,
            max_length_protein=self.config.max_length,
            add_special_tokens=False,
        ), EnsembleCollator({
            "protein": DataCollatorWithPadding(
                self.protein_tokenizer,
                padding=True
            )
        })

    def forward_protein_folding(self, protein: Featurized[Protein], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Training EsmFold is currently unavailable! Please see https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py for more details.")

    @torch.no_grad()
    def predict_protein_folding(self, 
        protein: Featurized[Protein], 
        **kwargs
    ) -> List[str]:
        device = protein.input_ids.device
        
        with self.maybe_autocast(device, dtype=torch.float16 if device != torch.device('cpu') and self.config.use_fp16_for_esm else torch.float32):
            output = self.protein_model(protein.input_ids)
        
        # TODO: convert output to open_biomed.data.Protein
        output = self.protein_model.output_to_pdb(output)
        self.temp_plddt_res = output[1]
        output = [Protein.from_pdb(item.split("\n")) for item in output[0]]

        return output
    