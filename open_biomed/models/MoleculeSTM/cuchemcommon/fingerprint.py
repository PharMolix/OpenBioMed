import logging
import os
from abc import ABC
from enum import Enum

import numpy as np
import pandas as pd
from cddd.inference import InferenceModel
from cuchem.utils.data_peddler import download_cddd_models
from rdkit import Chem
from rdkit.Chem import AllChem

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)


def calc_morgan_fingerprints(dataframe, smiles_col='canonical_smiles'):
    """Calculate Morgan fingerprints on SMILES strings

    Args:
        dataframe (pd.DataFrame): dataframe containing a SMILES column for calculation

    Returns:
        pd.DataFrame: new dataframe containing fingerprints
    """
    mf = MorganFingerprint()
    fp = mf.transform(dataframe, col_name=smiles_col)
    fp = pd.DataFrame(fp)
    fp.index = dataframe.index
    return fp


class TransformationDefaults(Enum):
    MorganFingerprint = {'radius': 2, 'nBits': 512}
    Embeddings = {}


class BaseTransformation(ABC):
    def __init__(self, **kwargs):
        self.name = None
        self.kwargs = None
        self.func = None

    def transform(self, data):
        return NotImplemented

    def transform_many(self, data):
        return list(map(self.transform, data))

    def __len__(self):
        return NotImplemented


class MorganFingerprint(BaseTransformation):

    def __init__(self, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        self.func = AllChem.GetMorganFingerprintAsBitVect

    def transform(self, data, col_name='transformed_smiles'):
        data = data[col_name]
        fp_array = []
        for mol in data:
            m = Chem.MolFromSmiles(mol)
            fp = self.func(m, **self.kwargs)
            fp_array.append(list(fp.ToBitString()))
        fp_array = np.asarray(fp_array)
        return fp_array

    def __len__(self):
        return self.kwargs['nBits']


class Embeddings(BaseTransformation):

    def __init__(self, use_gpu=True, cpu_threads=5, model_dir=None, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        model_dir = download_cddd_models()
        self.func = InferenceModel(model_dir, use_gpu=use_gpu, cpu_threads=cpu_threads)

    def transform(self, data):
        data = data['transformed_smiles']
        return self.func.seq_to_emb(data).squeeze()

    def inverse_transform(self, embeddings):
        "Embedding array -- individual compound embeddings are in rows"
        embeddings = np.asarray(embeddings)
        return self.func.emb_to_seq(embeddings)

    def __len__(self):
        return self.func.hparams.emb_size
