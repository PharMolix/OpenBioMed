# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import logging
# from abc import ABC
# from rdkit.Chem.SaltRemover import SaltRemover
# from cddd.preprocessing import remove_salt_stereo, filter_smiles

# logger = logging.getLogger(__name__)


# class BaseTransformation(ABC):
#     def __init__(self):
#         pass

#     def transform(self, data):
#         return NotImplemented

#     def transform_many(self, data):
#         return list(map(self.transform, data))
#         #return [self.filter(x) for x in data]


# class RemoveSalt(BaseTransformation):
#     def __init__(self, remover=SaltRemover()):
#         self.name = __class__.__name__.split('.')[-1]
#         self.remover = remover

#     def transform(self, data):
#         return remove_salt_stereo(data, self.remover)


# class PreprocessSmiles(BaseTransformation):
#     def __init__(self):
#         self.name = __class__.__name__.split('.')[-1]

#     def transform(self, data):
#         return filter_smiles(data)
