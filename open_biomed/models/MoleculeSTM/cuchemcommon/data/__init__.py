from typing import List


class ClusterWfDAO(object):
    """
    Base class for all DAO for fetching data for Clustering Workflows
    """

    def meta_df(self):
        """
        Returns df with dtype set for structure without any column filter.
        """
        return NotImplemented

    def fetch_molecular_embedding(self, n_molecules: int, cache_directory: str = None):
        """
        Fetch molecular properties from database/cache into a dask array.
        """
        return NotImplemented

    def fetch_molecular_embedding_by_id(self, molecule_id: List):
        """
        Fetch molecular properties from database for the given id. Id depends on
        the backend databse. For chemble DB it should be molregid.
        """
        return NotImplemented

    def fetch_id_from_smile(self, new_molecules: List):
        """
        Fetch molecular details for a list of molecules. The values in the list
        of molecules depends on database/service used. For e.g. it could be
        ChemblId or molreg_id for Chemble database.
        """
        return NotImplemented


class GenerativeWfDao(object):

    def fetch_id_from_chembl(self, id: List):
        """
        Fetch molecular details for a list of molecules. The values in the list
        of molecules depends on database/service used. For e.g. it could be
        ChemblId or molreg_id for Chemble database.
        """
        return NotImplemented
