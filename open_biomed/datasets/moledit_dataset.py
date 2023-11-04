import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import os
import csv

import torch
from torch.utils.data import Dataset
import pandas as pd

from feature.mol_featurizer import MolMultiModalFeaturizer
from feature.text_featurizer import TextTransformerTokFeaturizer
from utils.mol_utils import valid_smiles
from models.MoleculeSTM.models.mega_molbart.tokenizer import MolEncTokenizer

class MoleditDataset(Dataset, ABC):
    def __init__(self, path, config):
        super(MoleditDataset, self).__init__()
        self.path = path
        self.config = config
        self._load_data()
        self._featurize()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _featurize(self):
        featurizer = MolMultiModalFeaturizer(self.config)
        self.mols = [featurizer(smi) for smi in self.smiles]

        smiles_emb = [d['structure'] for d in self.mols]
        smiles_emb = [d['SMILES'] for d in smiles_emb]
        smiles_emb = [item for sublist in smiles_emb for item in sublist]
        tokens, orig_pad_masks = self._pad_seqs(smiles_emb)
        smiles = [{'input_ids': tokens, 'pad_masks': pad_masks} for tokens, pad_masks in zip(tokens, orig_pad_masks)]
        for i, dictionary in enumerate(smiles):
            self.mols[i]["structure"]["SMILES"] = dictionary


    @staticmethod
    def _pad_seqs(seqs, pad_token = 0):
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks

    def __getitem__(self, index):
        return self.mols[index]

    def __len__(self):
        return len(self.mols)


class ZINC250K(MoleditDataset):
    def __init__(self, path, config, split):
        self.split = split
        super(ZINC250K, self).__init__(path, config)

    def _load_data(self, subset_size=None):

        SMILES_file = os.path.join(self.path, "raw/250k_rndm_zinc_drugs_clean_3.csv")
        df = pd.read_csv(SMILES_file)
        smiles = df['smiles'].tolist()  # Already canonical SMILES
        self.smiles = [x.strip() for x in smiles]

        new_SMILES_file = os.path.join(self.path, "raw/smiles.csv")
        if not os.path.exists(new_SMILES_file):
            data_smiles_series = pd.Series(self.smiles)
            print("saving to {}".format(new_SMILES_file))
            data_smiles_series.to_csv(new_SMILES_file, index=False, header=False)

        if subset_size is not None:
            self.smiles = self.smiles[:subset_size]
        



SUPPORTED_MOLEDIT_DATASET = {
    "ZINC250K": ZINC250K
}

