from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import random
import torch
import os
from transformers import BatchEncoding, PreTrainedTokenizer

from open_biomed.data import Molecule, Pocket, Protein, Text

def sub_dict(in_dict: dict[str, Any], keys_to_include: List[str]) -> dict[str, Any]:
    # Get a sub-dictionary based on keys_to_include
    return {k: in_dict[k] for k in keys_to_include}

def sub_batch_by_interval(start: int, end: int, **kwargs: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    # Construct a smaller batch from [start, end) of the original batch
    new_batch = {}
    for key in kwargs:
        new_batch[key] = kwargs[key][start:end]
    return new_batch

def safe_index(l: List[Any], e: Any) -> int:
    # Return index of element e in list l. If e is not present, return the last index
    try:
        return l.index(e)
    except:
        return len(l) - 1

def concatenate_tokens(tokens_to_concat: List[BatchEncoding]) -> BatchEncoding:
    # Concatenate multiple tokenized results by putting the non-padding tokens together
    batch_size = tokens_to_concat[0].input_ids.shape[0]
    concatenated = {
        "input_ids": torch.cat([tokens.input_ids for tokens in tokens_to_concat], dim=-1),
        "attention_mask": torch.cat([tokens.attention_mask for tokens in tokens_to_concat], dim=-1),
    }
    non_padding_length = concatenated["attention_mask"].sum(-1)
    max_length = non_padding_length.max().item()

    new_input_ids = []
    new_attention_mask = []
    for i in range(batch_size):
        perm = torch.cat([
            torch.where(concatenated["attention_mask"][i] == 1)[0],  # non-padding tokens
            torch.where(concatenated["attention_mask"][i] == 0)[0],  # padding tokens
        ])
        new_input_ids.append(concatenated["input_ids"][i][perm[:max_length]])
        new_attention_mask.append(concatenated["attention_mask"][i][perm[:max_length]])

    return BatchEncoding({
        "input_ids": torch.stack(new_input_ids, dim=0),
        "attention_mask": torch.stack(new_attention_mask, dim=0),
    })

def collate_objects_as_list(inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    outputs = {}
    for sample in inputs:
        for k, v in sample.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)
    return outputs

def wrap_and_select_outputs(outputs: Any, context: Optional[TextIO]=None) -> Dict[str, Any]:
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, list):
        if isinstance(outputs[0], list):
            outputs = outputs[0]
        selected = random.randint(0, len(outputs) - 1)
        if len(outputs) > 1 and context is not None:
            context.write(f"Selected {selected}th output for downstream tools.\n")
        outputs = outputs[selected]
    if isinstance(outputs, tuple):
        wrapped_all = {}
        for out in outputs:
            wrapped = wrap_and_select_outputs(out)
            for key, value in wrapped.items():
                wrapped_all[key] = value
        return wrapped_all
    elif isinstance(outputs, Molecule):
        return {"molecule": outputs}
    elif isinstance(outputs, Protein):
        return {"protein": outputs}
    elif isinstance(outputs, Pocket):
        return {"pocket": outputs}
    elif isinstance(outputs, Text):
        return {"text": outputs}
    else:
        return {"output": outputs}

def create_tool_input(data_type: str, value: str) -> Any:
    if data_type == "molecule":
        if value.endswith(".sdf"):
            return Molecule.from_sdf_file(value)
        elif value.endswith(".pkl"):
            return Molecule.from_binary_file(value)
        else:
            return Molecule.from_smiles(value)
    elif data_type == "protein":
        if value.endswith(".pdb"):
            return Protein.from_pdb_file(value)
        if os.path.exists(value.replace(".pkl", ".pdb")):
            return Protein.from_pdb_file(value.replace(".pkl", ".pdb"))
        if value.endswith(".pkl"):
            return Protein.from_binary_file(value)
        else:
            return Protein.from_fasta(value)
    elif data_type == "pocket":
        return Pocket.from_binary_file(value)
    elif data_type == "text":
        return Text.from_str(value)
    else:
        return value
    