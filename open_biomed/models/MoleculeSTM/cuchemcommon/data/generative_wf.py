import logging
from typing import List

from cuchemcommon.data.helper.chembldata import ChEmblData
from cuchemcommon.utils.singleton import Singleton

from . import GenerativeWfDao

logger = logging.getLogger(__name__)


class ChemblGenerativeWfDao(GenerativeWfDao, metaclass=Singleton):

    def __init__(self, fp_type):
        self.chem_data = ChEmblData(fp_type)

    def fetch_id_from_chembl(self, id: List):
        logger.debug('Fetch ChEMBL ID using molregno...')
        return self.chem_data.fetch_id_from_chembl(id)
