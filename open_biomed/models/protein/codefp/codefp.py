from typing import Dict, List, Any, Tuple, Union, Optional
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
from .modeling_codefp import CodeFPFunctionalProteinDesignModel
from open_biomed.data.protein import Protein
from open_biomed.utils.config import Config
from open_biomed.models.task_models.go_guided_protein_generation import GoGuidedProteinGenerationModel
from open_biomed.utils.collator import Collator
from open_biomed.utils.featurizer import Featurizer
from torch.cuda.amp import autocast
import logging

log = logging.getLogger(__name__)



class GoTermsBatchCollator(Collator):
    def __call__(self, inputs: List[Any]) -> Any:
        return {
            "go_terms": [item["go_terms"] for item in inputs]
        }

class GOFeaturizer:
    def __init__(self, model_cfg: Config) -> None:
        self.device = torch.device("cpu")
        self.go_id_to_index = {}
        self.index_to_desc = {}
        self.cls_emb = {}
        
        # Mapping GO ID -> Index
        go_mapping_path = getattr(model_cfg, "go_mapping_path", "go_mapping.pkl")
        if os.path.exists(go_mapping_path):
            with open(go_mapping_path, 'rb') as f:
                self.go_id_to_index = pickle.load(f)
        else:
            raise FileNotFoundError(f"GO mapping file not found: {go_mapping_path}")
        
        # Mapping Index -> Description
        desc_mapping_path = getattr(model_cfg, "desc_mapping_path", "go_id_mapping.pkl")
        if os.path.exists(desc_mapping_path):
            with open(desc_mapping_path, 'rb') as f:
                desc2map_dict = pickle.load(f)
                if 'motif_desc_to_id' in desc2map_dict:
                    for desc, idx in desc2map_dict['motif_desc_to_id'].items():
                        self.index_to_desc[idx] = desc
        else:
            raise FileNotFoundError(f"GO description mapping file not found: {desc_mapping_path}")
        
        # Mapping Index -> Static Description   
        static_mapping_path = getattr(model_cfg, "static_mapping_path", "desc2map_dict_statics.pkl")
        if os.path.exists(static_mapping_path):
            with open(static_mapping_path, 'rb') as f:
                desc2map_dict_statics = pickle.load(f)
                for desc, idx in desc2map_dict_statics.items():
                    self.index_to_desc[idx] = desc
        else:
            raise FileNotFoundError(f"GO static description mapping file not found: {static_mapping_path}")
                    
        # GO Embeddings
        cls_emb_path = getattr(model_cfg, "cls_emb_path", "train_go_terms_cls_emb.pkl")
        if os.path.exists(cls_emb_path):
            with open(cls_emb_path, 'rb') as f:
                self.cls_emb = pickle.load(f)
        else:
            raise FileNotFoundError(f"GO embeddings file not found: {cls_emb_path}")
                
        self.go_num = getattr(model_cfg.cond, "go_num", 375) if hasattr(model_cfg, "cond") else 375

    def to(self, device: torch.device) -> "GOFeaturizer":
        self.device = device
        return self

    def standardize_desc(self, desc: str) -> str:
        if desc == "acetyl-CoA:L-glutamate N-acetyltransferase activity":
            return "L-glutamate N-acetyltransferase activity"
        if desc == "glutamate N-acetyltransferase activity":
            return "L-glutamate N-acetyltransferase activity, acting on acetyl-L-ornithine as donor"
        if desc == "methione N-acyltransferase activity":
            return "L-methionine N-acyltransferase activity"
        if desc == "oxidoreductase activity, acting on NAD(P)H, NAD(P) as acceptor":
            return "oxidoreductase activity, acting on NAD(P)H as acceptor"
        return desc

    def encode(self, go_terms: List[str]) -> Dict[str, Any]:
        go_f_mapped = []
        motif_struct_emb = torch.zeros(7, 1280, dtype=torch.float32, device=self.device)
        motif_num = 0
        
        for go_id in go_terms:
            if go_id in self.go_id_to_index:
                idx = self.go_id_to_index[go_id]
                go_f_mapped.append(idx)
                
                if motif_num < 7 and idx in self.index_to_desc:
                    desc = self.standardize_desc(self.index_to_desc[idx])
                    if desc in self.cls_emb:
                        raw_emb = self.cls_emb[desc]
                        try:
                            # raw_emb is list/array of vectors, we need mean
                            motif_struct_emb[motif_num] = torch.mean(torch.from_numpy(np.array(raw_emb)), dim=0).to(self.device)
                            motif_num += 1
                        except:
                            pass
            else:
                # Fallback to hash if not in mapping
                go_f_mapped.append(abs(hash(go_id)) % self.go_num)
                # logging warning
                logging.warning(f"GO ID {go_id} not found in mapping, using hash instead.")
        
        return {
            "go_label": torch.tensor(go_f_mapped, dtype=torch.long, device=self.device) if go_f_mapped else None,
            "motif_struct_emb": motif_struct_emb
        }


class GoTermsFeaturizer(Featurizer):
    def __call__(self, go_terms: Union[str, List[str]]) -> Dict[str, Any]:
        if isinstance(go_terms, str):
            go_terms = [go_terms]
        return {"go_terms": go_terms}

    def get_attrs(self) -> List[str]:
        return ["go_terms"]


class CodeFP(GoGuidedProteinGenerationModel):

    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        # print(model_cfg.todict())
        self.model = CodeFPFunctionalProteinDesignModel.from_pretrained(model_cfg['ckpt_path'])
        # exit()
        self.go_featurizer = GOFeaturizer(model_cfg)
        self._add_task()

    def _add_task(self) -> None:
        featurizer, collator = self.featurizer_go_guided_protein_generation()
        self.supported_tasks["go_guided_protein_generation"] = {
            "forward_fn": self.forward_go_guided_protein_generation,
            "predict_fn": self.predict_go_guided_protein_generation,
            "featurizer": featurizer,
            "collator": collator,
        }

    def featurizer_go_guided_protein_generation(self) -> Tuple[Featurizer, Collator]:
        return GoTermsFeaturizer(), GoTermsBatchCollator()
    
    def forward_go_guided_protein_generation(self,
        go_terms: List[List[str]],
        label: List[Protein],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @torch.no_grad()
    def predict_go_guided_protein_generation(self,
        go_terms: List[List[str]], 
    ) -> List[Protein]:

        # log.warning("CodeFP only support one protein generation for now.")

        self.model.eval()
        device = self.model.device
        self.go_featurizer.to(device)
        designed_protein = []

        # print(go_terms)

        for go_term in go_terms:
            feats = self.go_featurizer.encode(go_term)

            seq_lens = getattr(self.config, "seq_lens", [200, 400])
            length = np.random.randint(seq_lens[0], seq_lens[1] + 1)

            tokenizer = self.model.tokenizer
            seq_struct = tokenizer.struct_cls_token + (tokenizer.all_tokens[50] * length) + tokenizer.struct_eos_token
            seq_aa = tokenizer.aa_cls_token + ("A" * length) + tokenizer.aa_eos_token

            batch_struct = tokenizer.batch_encode_plus(
                [seq_struct],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            batch_aatype = tokenizer.batch_encode_plus(
                [seq_aa],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = torch.concat([batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1).to(device)

            batch = {
                "input_ids": input_ids,
                "go_label": feats["go_label"] if feats["go_label"] is not None else None,
            }
            if self.model.use_motif_struct_emb:
                batch["motif_struct_emb"] = feats["motif_struct_emb"].unsqueeze(0)

            # print(batch)

            sampling_strategy = getattr(self.config, "sampling_strategy", "annealing@2.0:1.0")
            max_iter = getattr(self.config, "max_iter", 500)
            use_only_struct = getattr(self.config, "use_only_struct", False)

            # print(sampling_strategy)
            # print(max_iter)
            # print(use_only_struct)

            with autocast():
                gen_outputs = self.model.generate(
                    batch=batch,
                    max_iter=max_iter,
                    sampling_strategy=sampling_strategy,
                    use_struct_only=use_only_struct,
                )
            
            # print(gen_outputs[2])
            # exit()
            
            output_tokens = gen_outputs[0]
            _, aatype_tokens = output_tokens.chunk(2, dim=-1)

            # print(aatype_tokens)
            
            if use_only_struct:
                raise NotImplementedError("use_only_struct is not supported yet.")
            output_results = list(
                map(
                    lambda s: "".join(s.split()),
                    tokenizer.batch_decode(
                        aatype_tokens, skip_special_tokens=True
                    ),
                )
            )
            seq = output_results[0]
            print(f"go_term: {go_term}, seq: {seq}")
            designed_protein.append(Protein.from_fasta(seq))
        # print(designed_protein)
        # exit()
        return designed_protein
