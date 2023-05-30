from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data, Batch
from transformers import BatchEncoding, DataCollatorWithPadding, BertTokenizer, T5Tokenizer, GPT2Tokenizer

name2tokenizer = {
    "bert": BertTokenizer,
    "t5": T5Tokenizer,
    "gpt2": GPT2Tokenizer
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
                result[key] = self._collate_single([x[key] for x in data], config[key])
            return result
        elif isinstance(data[0], int):
            return torch.tensor(data).view((-1, 1))
        elif isinstance(data[0], list):
            return [torch.tensor(i) for i in data]

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

class DrugCollator(BaseCollator):
    def __init__(self, config):
        super(DrugCollator, self).__init__(config)

    def __call__(self, drugs):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                batch[modality] = self._collate_single([drug[modality] for drug in drugs], self.config["featurizer"][modality])
        else:
            if isinstance(drugs[0], dict):
                drugs = [drug["structure"] for drug in drugs]
            batch = self._collate_single(drugs, self.config["featurizer"]["structure"])
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
        self.drug_collator = DrugCollator(config)

    def __call__(self, data):
        drugs, labels = map(list, zip(*data))
        return self.drug_collator(drugs), torch.stack(labels)

class DTICollator(TaskCollator):
    def __init__(self, config):
        super(DTICollator, self).__init__(config)
        self.drug_collator = DrugCollator(config["drug"])
        self.protein_collator = ProteinCollator(config["protein"])

    def __call__(self, data):
        drugs, prots, labels = map(list, zip(*data))
        return self.drug_collator(drugs), self.protein_collator(prots), torch.tensor(labels)

class DRPCollator(TaskCollator):
    def __init__(self, config):
        super(DRPCollator, self).__init__(config)
        self.drug_collator = DrugCollator(config["drug"])
        self.cell_collator = CellCollator(config["cell"])

    def __call__(self, data):
        drugs, cells, labels = map(list, zip(*data))
        return self.drug_collator(drugs), self.cell_collator(cells), torch.tensor(labels)