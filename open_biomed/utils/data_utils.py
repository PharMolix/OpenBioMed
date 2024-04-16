from open_biomed.feature.mol_featurizer import SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer
from open_biomed.feature.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from open_biomed.feature.cell_featurizer import SUPPORTED_CELL_FEATURIZER
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from open_biomed.utils.collators import MolCollator, ProteinCollator, CellCollator, TextCollator

entity_featurizer_map = {
    "molecule": (SUPPORTED_MOL_FEATURIZER, MolMultiModalFeaturizer),
    "protein": (SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer),
    "cell": (SUPPORTED_CELL_FEATURIZER, None),
    "text": (SUPPORTED_TEXT_FEATURIZER, None),
}

entity_collator_map = {
    "molecule": MolCollator,
    "protein": ProteinCollator,
    "cell": CellCollator,
    "text": TextCollator
}

class DataProcessorFast(object):
    def __init__(self, entity_type, config):
        self.entity_type = entity_type
        self.config = config
        assert self.entity_type in entity_featurizer_map
        # configure featurizer
        feat = entity_featurizer_map[self.entity_type]
        if entity_type in ["molecule", "protein"]:
            if len(self.config["modality"]) > 1:
                if feat[1] is None:
                    raise NotImplementedError("Multi-Modal featurizer for %s is not implemented!" % (self.entity_type))
                self.featurizer = feat[1](config)
            else:
                feat_config = self.config["featurizer"][self.config["modality"][0]]
                if feat_config["name"] not in feat[0]:
                    raise NotImplementedError("Featurizer %s for %s is not implemented!" % (feat_config["name"], self.entity_type))
                self.featurizer = feat[0][feat_config["name"]](feat_config)
        else:
            self.featurizer = feat[0][config["name"]](config)
        # configure collator
        self.collator = entity_collator_map[self.entity_type](self.config)

    def __call__(self, obj):
        if not isinstance(obj, list):
            obj = [self.featurizer(obj)]
        else:
            obj = [self.featurizer(x) for x in obj]
        return self.collator(obj)
        