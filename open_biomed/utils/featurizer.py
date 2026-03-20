from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import BatchEncoding, T5Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from open_biomed.data import Molecule, Protein, Pocket, Text

T = TypeVar('T', bound=Any)

class Featurized(Generic[T]):
    def __init__(self, value: T):
        self.value = value

class Featurizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_attrs(self) -> List[str]:
        raise NotImplementedError

class IdenticalFeaturizer(Featurizer):
    def __init__(self, attrs: List[str]) -> None:
        super().__init__()
        self.attrs = attrs

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return args[0]

    def get_attrs(self) -> List[str]:
        return self.attrs

class MoleculeFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["molecule"]

class MoleculeTransformersFeaturizer(MoleculeFeaturizer):
    def __init__(self,
        tokenizer: PreTrainedTokenizer,
        max_length: int=512,
        add_special_tokens: bool=True,
        base: str='SMILES',
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.base = base
        if base not in ["SMILES", "SELFIES"]:
            raise ValueError(f"{base} is not a valid 1D representaiton of molecules!")

    def __call__(self, molecule: Molecule) -> BatchEncoding:
        if self.base == "SMILES":
            molecule._add_smiles()
            parse_str = molecule.smiles
        if self.base == "SELFIES":
            molecule._add_selfies()
            parse_str = molecule.selfies
        return self.tokenizer(
            parse_str, 
            max_length=self.max_length,
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )

class TextFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, text: Text) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["text"]

class TextTransformersFeaturizer(TextFeaturizer):
    def __init__(self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int=512,
        add_special_tokens: bool=True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __call__(self, text: Text) -> BatchEncoding:
        return self.tokenizer(
            text.str, 
            max_length=self.max_length,
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )

class ProteinFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, protein: Protein) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["protein"]

class ProteinTransformersFeaturizer(ProteinFeaturizer):
    def __init__(self,
        tokenizer: PreTrainedTokenizer,
        max_length: int=1024,
        add_special_tokens: bool=True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __call__(self, protein: Protein) -> Dict[str, Any]:
        return self.tokenizer(
            protein.sequence,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        )

class PocketFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, pocket: Pocket) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["pocket"]

# For classification tasks, directly convert numbers or arrays into tensors.
class ClassLabelFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    # Input a number or an array, and return a tensor.
    def __call__(self, label: np.array) -> torch.tensor:
        return  torch.tensor(label)

    def get_attrs(self) -> List[str]:
        return ["classlabel"]


class EnsembleFeaturizer(Featurizer):
    def __init__(self, to_ensemble: Dict[str, Featurizer]) -> None:
        super().__init__()
        self.featurizers = {}
        for k, v in to_ensemble.items():
            self.featurizers[k] = v

    def __call__(self, **kwargs: Any) -> Dict[Any, Any]:
        featurized = {}
        for k, v in kwargs.items():
            featurized[k] = self.featurizers[k](v)
        return featurized

    def get_attrs(self) -> List[str]:
        return list(self.featurizers.keys())

if __name__ == "__main__":
    from transformers import DataCollatorWithPadding
    featurizer = TextTransformersFeaturizer(T5Tokenizer.from_pretrained("./checkpoints/molt5/base"))
    a = featurizer(text=Text.from_str("Hello"))
    b = featurizer(text=Text.from_str("Hello World"))
    collator = DataCollatorWithPadding(featurizer.tokenizer, padding=True)
    print(collator([a, b]))