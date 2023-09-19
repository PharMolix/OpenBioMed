from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data, Batch
from transformers import BatchEncoding, DataCollatorWithPadding, BertTokenizer, T5Tokenizer, GPT2Tokenizer, EsmTokenizer
from utils.mol_utils import SmilesTokenizer

name2tokenizer = {
    "bert": BertTokenizer,
    "t5": T5Tokenizer,
    "gpt2": GPT2Tokenizer,
    "esm": EsmTokenizer,
    "unimap": SmilesTokenizer,
}

def ToDevice(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple) or isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    else:
        return obj.to(device)

class BaseCollator(ABC):
    def __init__(self, config):
        self.config = config
        self._build(config)

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError

    def _collate_single(self, data, config):
        if isinstance(data[0], Data):
            return Batch.from_data_list(data)
        elif torch.is_tensor(data[0]):
            return torch.stack([x.squeeze() for x in data])
        elif isinstance(data[0], BatchEncoding):
            return config["collator"](data)
        elif isinstance(data[0], dict):
            result = {}
            for key in data[0]:
                result[key] = self._collate_single([x[key] for x in data], config[key] if key in config else {})
            return result
        elif isinstance(data[0], int):
            return torch.tensor(data).view((-1, 1))

    def _collate_multiple(self, data, config):
        cor = []
        flatten_data = []
        for x in data:
            cor.append(len(flatten_data))
            flatten_data += x
        cor.append(len(flatten_data))
        return (cor, self._collate_single(flatten_data, config),)

    def _build(self, config):
        if not isinstance(config, dict):
            return
        if "model_name_or_path" in config:
            tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"])
            if config["transformer_type"] == "gpt2":
                tokenizer.pad_token = tokenizer.eos_token
            config["collator"] = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True
            )
            return
        for key in config:
            self._build(config[key])

class MolCollator(BaseCollator):
    def __init__(self, config):
        super(MolCollator, self).__init__(config)

    def __call__(self, mols):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                batch[modality] = self._collate_single([mol[modality] for mol in mols], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(mols, self.config["featurizer"]["structure"])
        return batch

class ProteinCollator(BaseCollator):
    def __init__(self, config):
        super(ProteinCollator, self).__init__(config)

    def __call__(self, proteins):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                if isinstance(proteins[0][modality], list):
                    batch[modality] = self._collate_multiple([protein[modality] for protein in proteins], self.config["featurizer"][modality])
                else:
                    batch[modality] = self._collate_single([protein[modality] for protein in proteins], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(proteins, self.config["featurizer"]["structure"])
        return batch

class CellCollator(BaseCollator):
    def __init__(self, config):
        super(CellCollator, self).__init__(config)

    def __call__(self, cells):
        batch = self._collate_single(cells, self.config["featurizer"])
        return batch

class TextCollator(BaseCollator):
    def __init__(self, config):
        super(TextCollator, self).__init__(config)

    def __call__(self, texts):
        batch = self._collate_single(texts, self.config)
        return batch

class TaskCollator(ABC):
    def __init__(self, config):
        super(TaskCollator, self).__init__()
        self.config = config

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError

class DPCollator(TaskCollator):
    def __init__(self, config):
        super(DPCollator, self).__init__(config)
        self.mol_collator = MolCollator(config)

    def __call__(self, data):
        mols, labels = map(list, zip(*data))
        return self.mol_collator(mols), torch.stack(labels)

class DDICollator(TaskCollator):
    def __init__(self, config):
        super(DDICollator, self).__init__(config)
        self.mol_collator = MolCollator(config['mol'])

    def __call__(self, data):
        molA, molB, labels = map(list, zip(*data))
        return self.mol_collator(molA), self.mol_collator(molB), torch.tensor(labels)

class DTICollator(TaskCollator):
    def __init__(self, config):
        super(DTICollator, self).__init__(config)
        self.mol_collator = MolCollator(config["mol"])
        self.protein_collator = ProteinCollator(config["protein"])

    def __call__(self, data):
        mols, prots, labels = map(list, zip(*data))
        return self.mol_collator(mols), self.protein_collator(prots), torch.tensor(labels)

class DRPCollator(TaskCollator):
    def __init__(self, config):
        super(DRPCollator, self).__init__(config)
        self.mol_collator = MolCollator(config["mol"])
        self.cell_collator = CellCollator(config["cell"])

    def __call__(self, data):
        mols, cells, labels = map(list, zip(*data))
        return self.mol_collator(mols), self.cell_collator(cells), torch.tensor(labels)

class PPICollator(TaskCollator):
    def __init__(self, config, graph_ppi):
        super(PPICollator, self).__init__(config)
        self.graph_ppi = graph_ppi
        self.protein_collator = ProteinCollator(config)

    def __call__(self, data):
        prots1, prots2, labels = map(list, zip(*data))
        if self.graph_ppi:
            return torch.LongTensor(prots1), torch.LongTensor(prots2), torch.stack(labels)
        else:
            return self.protein_collator(prots1), self.protein_collator(prots2), torch.stack(labels)

class MolQACollator(TaskCollator):
    def __init__(self, config, collate_outputs=True):
        super(MolQACollator, self).__init__(config)
        self.mol_collator = MolCollator(config["mol"])
        self.question_collator = TextCollator(config["text"]["question"])
        self.collate_outputs = collate_outputs
        if self.collate_outputs:
            self.answer_collator = TextCollator(config["text"]["answer"])

    def  __call__(self, data):
        mols, questions, answers = map(list, zip(*data))
        if self.collate_outputs:
            return self.mol_collator(mols), self.question_collator(questions), self.answer_collator(answers)
        else:
            return self.mol_collator(mols), self.question_collator(questions), answers