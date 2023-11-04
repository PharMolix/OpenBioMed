import random
from collections import defaultdict
from itertools import compress

import numpy as np
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Subset


def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold
    

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   pyg_dataset=True):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
    :param frac_train, frac_valid, frac_test: fractions
    :param pyg_dataset: if this is pytorch or pytorch-gemetric dataset
    :return: train, valid, test slices of the input dataset obj. """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    if pyg_dataset:
        train_dataset = dataset[torch.tensor(train_idx)]
        valid_dataset = dataset[torch.tensor(valid_idx)]
        test_dataset = dataset[torch.tensor(test_idx)]
        return train_dataset, valid_dataset, test_dataset
    else:
        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)
        return train_dataset, valid_dataset, test_dataset
