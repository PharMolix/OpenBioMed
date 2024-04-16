from abc import ABC, abstractmethod

from scipy.spatial import distance_matrix
import torch
from torch_geometric.data import Data, Batch
from transformers import BatchEncoding, DataCollatorWithPadding, BertTokenizer, T5Tokenizer, GPT2Tokenizer, EsmTokenizer, LlamaTokenizer
from open_biomed.utils.mol_utils import SmilesTokenizer, get_unimol_dictionary

name2tokenizer = {
    "bert": BertTokenizer,
    "geneformer": BertTokenizer,
    "biot5": T5Tokenizer,
    "t5": T5Tokenizer,
    "gpt2": GPT2Tokenizer,
    "esm": EsmTokenizer,
    "unimap": SmilesTokenizer,
    "llama": LlamaTokenizer,
    "3d-molm": LlamaTokenizer,
}

def ToDevice(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    elif isinstance(obj, tuple):
        ret = [ToDevice(x, device) for x in obj]
        return tuple(ret)
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
        if "transformer_type" in config:
            tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"])
            if config["transformer_type"] == "gpt2":
                tokenizer.pad_token = tokenizer.eos_token
            if config["transformer_type"] in ["3d-molm", "llama"]:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.add_special_tokens({'bos_token': '</s>'})
                tokenizer.add_special_tokens({'eos_token': '</s>'})
                tokenizer.add_special_tokens({'unk_token': '</s>'})
            if config["transformer_type"] == "3d-molm":
                tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
            config["collator"] = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True
            )
            return
        for key in config:
            self._build(config[key])

class UniMolCollator(BaseCollator):
    def __init__(self, config):
        super(UniMolCollator, self).__init__(config)
        self.dictionary = get_unimol_dictionary(config["dictionary_path"])

    def _pad_1d_tokens(
        self,
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
    ):
        """
        padding one dimension tokens inputs.

        :param values: A list of 1d tensors.
        :param pad_idx: The padding index.
        :param left_pad: Whether to left pad the tensors. Defaults to False.
        :param pad_to_length: The desired length of the padded tensors. Defaults to None.
        :param pad_to_multiple: The multiple to pad the tensors to. Defaults to 1.

        :return: A padded 1d tensor as a torch.Tensor.

        """
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
        return res


    def _pad_2d(
        self,
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
    ):
        """
        padding two dimension tensor inputs.

        :param values: A list of 2d tensors.
        :param pad_idx: The padding index.
        :param left_pad: Whether to pad on the left side. Defaults to False.
        :param pad_to_length: The length to pad the tensors to. If None, the maximum length in the list
                            is used. Defaults to None.
        :param pad_to_multiple: The multiple to pad the tensors to. Defaults to 1.

        :return: A padded 2d tensor as a torch.Tensor.
        """
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = values[0].new(len(values), size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :, size - len(v) :] if left_pad else res[i][: len(v), : len(v)])
        return res

    def __call__(self, mols):
        return (
            self._pad_1d_tokens([m['atoms'] for m in mols], self.dictionary.pad(), pad_to_length=256),
            self._pad_2d([torch.tensor(distance_matrix(m['coordinates'][0], m['coordinates'][0])).float() for m in mols], 0.0, pad_to_length=256),
            self._pad_2d([m['atoms'].view(-1, 1) * len(self.dictionary) + m['atoms'].view(1, -1) for m in mols], self.dictionary.pad(), pad_to_length=256)
        )

class MolCollator(BaseCollator):
    def __init__(self, config):
        super(MolCollator, self).__init__(config)
        if self.config["featurizer"]["structure"]["name"] == "unimol":
            self.structure_collator = UniMolCollator(config["featurizer"]["structure"])
        elif self.config["featurizer"]["structure"]["name"] == "MultiScale" and "conformation" in self.config["featurizer"]["structure"]["scales"]:
            self.structure_collator = UniMolCollator(config["featurizer"]["structure"]["conformation"])
        elif self.config["featurizer"]["structure"]["name"] == "Ensemble" and "unimol" in self.config["featurizer"]["structure"]["models"]:
            self.structure_collator = UniMolCollator(config["featurizer"]["structure"]["unimol"])
        else:
            self.structure_collator = None

    def __call__(self, mols):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                if modality == "structure" and self.structure_collator is not None:
                    if self.config["featurizer"]["structure"]["name"] == "MultiScale":
                        processed = {}
                        for scale in self.config["featurizer"]["structure"]["scales"]:
                            if scale != "conformation":
                                processed[scale] = self._collate_single([mol[modality][scale] for mol in mols], self.config["featurizer"][modality][scale])
                            else:
                                processed[scale] = self.structure_collator([mol[modality][scale] for mol in mols])
                        batch[modality] = processed
                    elif self.config["featurizer"]["structure"]["name"] == "Ensemble":
                        processed = {}
                        for model in self.config["featurizer"]["structure"]["models"]:
                            if model != "unimol":
                                processed[model] = self._collate_single([mol[modality][model] for mol in mols], self.config["featurizer"][modality][model])
                            else:
                                processed[model] = self.structure_collator([mol[modality][model] for mol in mols])
                        batch[modality] = processed
                    else:
                        batch[modality] = self.structure_collator([mol[modality] for mol in mols])
                else:
                    batch[modality] = self._collate_single([mol[modality] for mol in mols], self.config["featurizer"][modality])
        elif self.config["featurizer"]["structure"]["name"] == "MultiScale":
            batch = {}
            for scale in self.config["featurizer"]["structure"]["scales"]:
                if scale != "conformation":
                    batch[scale] = self._collate_single([mol[scale] for mol in mols], self.config["featurizer"]["structure"][scale])
                else:
                    batch[scale] = self.structure_collator([mol[scale] for mol in mols])
        else:
            if self.structure_collator is None:
                batch = self._collate_single(mols, self.config["featurizer"]["structure"])
            else:
                batch = self.structure_collator(mols)
        return batch

class MultiMolCollator(BaseCollator):
    def __init__(self, config):
        super(MultiMolCollator, self).__init__(config)
        self.mol_collator = MolCollator(config)

    def __call__(self, mols):
        batch_idx = []
        mols_flatten = []
        for i, mol_list in enumerate(mols):
            mols_flatten += mol_list
            batch_idx += [i] * len(mol_list)
        return self.mol_collator(mols_flatten), torch.LongTensor(batch_idx)

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

class MultiProteinCollator(BaseCollator):
    def __init__(self, config):
        super(MultiProteinCollator, self).__init__(config)
        self.protein_collator = ProteinCollator(config)

    def __call__(self, prots):
        batch_idx = []
        prots_flatten = []
        for i, prot_list in enumerate(prots):
            prots_flatten += prot_list
            batch_idx += [i] * len(prot_list)
        return self.protein_collator(prots_flatten), torch.LongTensor(batch_idx)

class CellCollator(BaseCollator):
    def __init__(self, config):
        super(CellCollator, self).__init__(config)

    def __call__(self, cells):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                batch[modality] = self._collate_single([cell[modality] for cell in cells], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(cells, self.config["featurizer"]["structure"])
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

class MTCollator(TaskCollator):
    def __init__(self, config):
        super(MTCollator, self).__init__(config)
        self.mol_collator = MolCollator(config["mol"])
        self.text_collator = TextCollator(config["text"])

    def __call__(self, data):
        mols, texts = map(list, zip(*data))
        return self.mol_collator(mols), self.text_collator(texts)

class MolQACollator(TaskCollator):
    def __init__(self, config, collate_outputs=True):
        super(MolQACollator, self).__init__(config)
        self.mol_collator = MultiMolCollator(config["mol"])
        self.question_collator = TextCollator(config["text"]["question"])
        self.collate_outputs = collate_outputs
        if self.collate_outputs:
            self.answer_collator = TextCollator(config["text"]["answer"])

    def  __call__(self, data):
        mols, questions, answers = map(list, zip(*data))
        mols, batch = self.mol_collator(mols)
        if self.collate_outputs:
            return mols, batch, self.question_collator(questions), self.answer_collator(answers)
        else:
            return mols, batch, self.question_collator(questions), answers

class ProteinQACollator(TaskCollator):
    def __init__(self, config, collate_outputs=True):
        super(ProteinQACollator, self).__init__(config)
        self.mol_collator = MultiProteinCollator(config["protein"])
        self.question_collator = TextCollator(config["text"]["question"])
        self.collate_outputs = collate_outputs
        if self.collate_outputs:
            self.answer_collator = TextCollator(config["text"]["answer"])

    def  __call__(self, data):
        proteins, questions, answers = map(list, zip(*data))
        proteins, batch = self.mol_collator(proteins)
        if self.collate_outputs:
            return proteins, batch, self.question_collator(questions), self.answer_collator(answers)
        else:
            return proteins, batch, self.question_collator(questions), answers

class CellQACollator(TaskCollator):
    def __init__(self, config, collate_outputs=True):
        super(CellQACollator, self).__init__(config)
        self.cell_collator = CellCollator(config["cell"])
        self.question_collator = TextCollator(config["text"]["question"])
        self.collate_outputs = collate_outputs
        if self.collate_outputs:
            self.answer_collator = TextCollator(config["text"]["answer"])

    def  __call__(self, data):
        cells, questions, answers = map(list, zip(*data))
        if self.collate_outputs:
            return self.cell_collator(cells), self.question_collator(questions), self.answer_collator(answers)
        else:
            return self.cell_collator(cells), self.question_collator(questions), answers