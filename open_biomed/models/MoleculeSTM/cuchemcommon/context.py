import logging
import os
from configparser import RawConfigParser
from io import StringIO

from models.MoleculeSTM.cuchemcommon.utils.singleton import Singleton

logger = logging.getLogger(__name__)

CONFIG_FILE = '.env'


class Context(metaclass=Singleton):

    def __init__(self):

        self.dask_client = None
        self.compute_type = 'gpu'
        self.is_benchmark = False
        self.benchmark_file = None
        self.cache_directory = None
        self.n_molecule = None
        self.batch_size = 10000

        self.config = {}
        if os.path.exists(CONFIG_FILE):
            logger.info('Reading properties from %s...', CONFIG_FILE)
            self.config = self._load_properties_file(CONFIG_FILE)
        else:
            logger.warn('Could not locate %s', CONFIG_FILE)

    def _load_properties_file(self, properties_file):
        """
        Reads a properties file using ConfigParser.

        :param propertiesFile/configFile:
        """
        config_file = open(properties_file, 'r')
        config_content = StringIO('[root]\n' + config_file.read())
        config = RawConfigParser()
        config.read_file(config_content)

        return config._sections['root']

    def get_config(self, config_name, default=None):
        """
        Returns values from local configuration.
        """
        try:
            return self.config[config_name]
        except KeyError:
            logger.warn('%s not found, returing default.', config_name)
            return default
