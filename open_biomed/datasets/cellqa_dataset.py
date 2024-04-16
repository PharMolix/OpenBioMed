import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import os
import json
import scanpy
from transformers import BatchEncoding
import torch
from torch.utils.data import Dataset

from open_biomed.feature.cell_featurizer import SUPPORTED_CELL_FEATURIZER
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER

class CellQADataset(Dataset, ABC):
    def __init__(self, path, config):
        super(CellQADataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize(True if self.split == "train" else False)

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self, featurize_output=True):
        featurizer = SUPPORTED_CELL_FEATURIZER[self.config["cell"]["featurizer"]["structure"]["name"]](self.config["cell"]["featurizer"]["structure"])
        if self.config["cell"]["featurizer"]["structure"]["name"] == "GeneFormer":
            self.cells = featurizer(self.cells)
        else:
            self.cells = [featurizer(cell) for cell in self.cells]
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["question"]["name"]](self.config["text"]["question"])
        self.questions = [featurizer(text) for text in self.questions]
        if featurize_output:
            featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["answer"]["name"]](self.config["text"]["answer"])
            self.answers = [featurizer(text) for text in self.answers]

class CQA(CellQADataset):
    def __init__(self, path, config, split):
        self.split = split
        super(CQA, self).__init__(path, config)

    def _load_data(self):
        self.data = json.load(open(os.path.join(self.path, "processed", self.split + ".json")))
        self.all_files = set()
        for data in self.data:
            if data["group"] not in self.all_files:
                self.all_files.add(data["group"])
        
    def _featurize(self, featurize_output=True):
        self.cells = []
        self.id2cellmap = {}
        featurizer = SUPPORTED_CELL_FEATURIZER[self.config["cell"]["featurizer"]["structure"]["name"]](self.config["cell"]["featurizer"]["structure"])
        for file in self.all_files:
            self.id2cellmap[file] = {}
            data = scanpy.read_h5ad(os.path.join(self.path, "h5ads", file + ".h5ad"))
            N = len(self.cells)
            for i, id in enumerate(data.obs_names):
                self.id2cellmap[file][id] = N + i
            if data.raw is not None:
                data.X = data.raw.X
            if self.config["cell"]["featurizer"]["structure"]["name"] == "GeneFormer":
                input_ids, _ = featurizer(data)
                self.cells += [BatchEncoding({"input_ids": torch.LongTensor(i), "attention_mask": torch.ones(len(i))}) for i in input_ids]
            else:
                for i in range(len(data.X)):
                    self.cells.append(featurizer(data.X[i]))
        featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["question"]["name"]](self.config["text"]["question"])
        for i in range(len(self.data)):
            self.data[i]["question"] = featurizer(self.data[i]["question"])
        if featurize_output:
            featurizer = SUPPORTED_TEXT_FEATURIZER[self.config["text"]["answer"]["name"]](self.config["text"]["answer"])
            for i in range(len(self.data)):
                self.data[i]["answer"] = featurizer(self.data[i]["answer"])

    def __getitem__(self, index):
        data = self.data[index]
        return self.cells[self.id2cellmap[data["group"]][data["id"]]], data["question"], data["answer"]

    def __len__(self):
        return len(self.data)

SUPPORTED_CELLQA_DATASET = {
    "cqa": CQA,
}