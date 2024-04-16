from abc import ABC, abstractmethod
import pickle
import os

class BaseFeaturizer(ABC):
    def __init__(self, config=None):
        super(BaseFeaturizer, self).__init__()
        self.allow_cache = False if config is None or "allow_cache" not in config else config["allow_cache"]
        self.cache_file = config["cache_file"] if self.allow_cache else ""
        if self.allow_cache:
            self.cached_data = {}
            if os.path.exists(self.cache_file):
                self.cached_data = pickle.load(open(self.cache_file, "rb"))
    
    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

    def __del__(self):
        if self.allow_cache and not os.path.exists(self.cache_file):
            pickle.dump(self.cached_data, open(self.cache_file, "wb"))