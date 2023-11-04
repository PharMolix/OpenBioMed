import logging
import math
import os
from typing import List

import cudf
import dask
import dask_cudf
from cuchemcommon.context import Context
from cuchemcommon.data.helper.chembldata import BATCH_SIZE, ChEmblData
from cuchemcommon.utils.singleton import Singleton

from . import ClusterWfDAO

logger = logging.getLogger(__name__)

FINGER_PRINT_FILES = 'filter_*.h5'


class ChemblClusterWfDao(ClusterWfDAO, metaclass=Singleton):

    def __init__(self, fp_type):
        self.chem_data = ChEmblData(fp_type)

    def meta_df(self):
        chem_data = ChEmblData()
        return chem_data._meta_df()

    def fetch_molecular_embedding(self,
                                  n_molecules: int,
                                  cache_directory: str = None):
        context = Context()
        if cache_directory:
            hdf_path = os.path.join(cache_directory, FINGER_PRINT_FILES)
            logger.info('Reading %d rows from %s...', n_molecules, hdf_path)
            mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')

            if n_molecules > 0:
                npartitions = math.ceil(n_molecules / BATCH_SIZE)
                mol_df = mol_df.head(n_molecules, compute=False, npartitions=npartitions)
        else:
            logger.info('Reading molecules from database...')
            mol_df = self.chem_data.fetch_mol_embedding(num_recs=n_molecules,
                                                        batch_size=context.batch_size)

        return mol_df

    def fetch_molecular_embedding_by_id(self, molecule_id: List):
        context = Context()
        meta = self.chem_data._meta_df()
        fp_df = self.chem_data._fetch_mol_embedding(molregnos=molecule_id,
                                                    batch_size=context.batch_size) \
            .astype(meta.dtypes)

        fp_df = cudf.from_pandas(fp_df)
        fp_df = dask_cudf.from_cudf(fp_df, npartitions=1).reset_index()
        return fp_df

    def fetch_id_from_chembl(self, new_molecules: List):
        logger.debug('Fetch ChEMBL ID using molregno...')
        return self.chem_data.fetch_id_from_chembl(new_molecules)
