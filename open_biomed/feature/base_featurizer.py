from abc import ABC, abstractmethod

class BaseFeaturizer(ABC):
    def __init__(self):
        super(BaseFeaturizer, self).__init__()
    
    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError