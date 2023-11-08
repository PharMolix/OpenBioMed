import logging
# import torch
from functools import singledispatch
from typing import List

import numpy as np
from rdkit.Chem import PandasTools, CanonSmiles

logger = logging.getLogger(__name__)


@singledispatch
def add_jitter(embedding, radius, cnt, shape):
    return NotImplemented


@add_jitter.register(np.ndarray)
def _(embedding, radius, cnt, shape):

    distorteds = []
    for i in range(cnt):
        noise = np.random.normal(0, radius, embedding.shape)
        distorted = noise + embedding
        distorteds.append(distorted)

    return distorteds

class GenerativeWfDao(object):

    def fetch_id_from_chembl(self, id: List):
        """
        Fetch molecular details for a list of molecules. The values in the list
        of molecules depends on database/service used. For e.g. it could be
        ChemblId or molreg_id for Chemble database.
        """
        return NotImplemented


class BaseGenerativeWorkflow:

    def __init__(self, dao: GenerativeWfDao = None) -> None:
        self.dao = dao
        self.min_jitter_radius = None

    def get_iteration(self):
        NotImplemented

    def smiles_to_embedding(self,
                            smiles: str,
                            padding: int):
        NotImplemented

    def embedding_to_smiles(self,
                            embedding: float,
                            dim: int,
                            pad_mask):
        NotImplemented

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False):
        NotImplemented

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False):
        NotImplemented

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False):
        NotImplemented

    def _compute_radius(self, scaled_radius):
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def addjitter(self,
                  embedding,
                  radius=None,
                  cnt=1,
                  shape=None):
        radius = radius if radius else self.radius_scale
        return add_jitter(embedding, radius, cnt, shape)

    def compute_unique_smiles(self,
                              interp_df,
                              embedding_funct,
                              scaled_radius=None):
        """
        Identify duplicate SMILES and distorts the embedding. The input df
        must have columns 'SMILES' and 'Generated' at 0th and 1st position.
        'Generated' colunm must contain boolean to classify SMILES into input
        SMILES(False) and generated SMILES(True).

        This function does not make any assumptions about order of embeddings.
        Instead it simply orders the df by SMILES to identify the duplicates.
        """

        distance = self._compute_radius(scaled_radius)
        embeddings = interp_df['embeddings']
        embeddings_dim = interp_df['embeddings_dim']
        for index, row in interp_df.iterrows():
            smile_string = row['SMILES']
            try:
                canonical_smile = CanonSmiles(smile_string)
            except:
                # If a SMILES cannot be canonicalized, just use the original
                canonical_smile = smile_string

            row['SMILES'] = canonical_smile

        for i in range(5):
            smiles = interp_df['SMILES'].sort_values()
            duplicates = set()
            for idx in range(0, smiles.shape[0] - 1):
                if smiles.iat[idx] == smiles.iat[idx + 1]:
                    duplicates.add(smiles.index[idx])
                    duplicates.add(smiles.index[idx + 1])

            if len(duplicates) > 0:
                for dup_idx in duplicates:
                    if interp_df.iat[dup_idx, 3]:
                        # add jitter to generated molecules only
                        distored = self.addjitter(embeddings[dup_idx],
                                                  distance,
                                                  cnt=1,
                                                  shape=embeddings_dim[dup_idx])
                        embeddings[dup_idx] = distored[0]
                interp_df['SMILES'] = embedding_funct(embeddings.to_list())
                interp_df['embeddings'] = embeddings
            else:
                break

        # Ensure all generated molecules are valid.
        for i in range(5):
            PandasTools.AddMoleculeColumnToFrame(interp_df, 'SMILES')
            invalid_mol_df = interp_df[interp_df['ROMol'].isnull()]

            if not invalid_mol_df.empty:
                invalid_index = invalid_mol_df.index.to_list()
                for idx in invalid_index:
                    embeddings[idx] = self.addjitter(embeddings[idx],
                                                     distance,
                                                     cnt=1,
                                                     shape=embeddings_dim[idx])[0]
                interp_df['SMILES'] = embedding_funct(embeddings.to_list())
                interp_df['embeddings'] = embeddings
            else:
                break

        # Cleanup
        if 'ROMol' in interp_df.columns:
            interp_df = interp_df.drop('ROMol', axis=1)

        return interp_df

    def interpolate_by_id(self,
                          ids: List,
                          id_type: str = 'chembleid',
                          num_points=10,
                          force_unique=False,
                          scaled_radius: int = 1):
        smiles = None

        if not self.min_jitter_radius:
            raise Exception('Property `radius_scale` must be defined in model class.')

        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(ids)]
            if len(smiles) != len(ids):
                raise Exception('One of the ids is invalid %s', ids)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.interpolate_smiles(smiles,
                                       num_points=num_points,
                                       scaled_radius=scaled_radius,
                                       force_unique=force_unique)

    def find_similars_smiles_by_id(self,
                                   chemble_id: str,
                                   id_type: str = 'chembleid',
                                   num_requested=10,
                                   force_unique=False,
                                   scaled_radius: int = 1):
        smiles = None

        if not self.min_jitter_radius:
            raise Exception('Property `radius_scale` must be defined in model class.')

        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(chemble_id)]
            if len(smiles) != len(chemble_id):
                raise Exception('One of the ids is invalid %s' + chemble_id)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.find_similars_smiles(smiles[0],
                                         num_requested=num_requested,
                                         scaled_radius=scaled_radius,
                                         force_unique=force_unique)
