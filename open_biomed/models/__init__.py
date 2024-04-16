from open_biomed.models.molecule import *
from open_biomed.models.protein import *
from open_biomed.models.cell import *
from open_biomed.models.knowledge import *
from open_biomed.models.text import *
from open_biomed.models.multimodal import *

SUPPORTED_MOL_ENCODER = {
    "cnn": MolCNN,
    "tgsa": GINTGSA,
    "graphcl": GraphCL,
    "graphmvp": GraphMVP,
    "molclr": MolCLR,
    "mgnn": MGNN,
    "molt5": MolT5,
    "bert": MolBERT,
    "biomedgpt-1.6b": BioMedGPTCLIP,
    "biomedgpt-10b": BioMedGPTV,
    "kv-plm": KVPLM,
    "momu": MoMu,
    "molfm": MolFM
}

SUPPORTED_MOL_DECODER = {
    "moflow": MoFlow,
    "molt5": MolT5
}

SUPPORTED_PROTEIN_ENCODER = {
    "cnn": ProtCNN,
    "cnn_gru": CNNGRU,
    "mcnn": MCNN,
    "pipr": CNNPIPR,
    "prottrans": ProtTrans
}

SUPPORTED_CELL_ENCODER = {
    "scbert": PerformerLM,
    "celllm": PerformerLM_CellLM,
    "geneformer": GeneFormer,
}

SUPPORTED_TEXT_ENCODER = {
    "base_transformer": BaseTransformers,
    "biomedgpt-1.6b": BioMedGPTCLIP,
    "biomedgpt-10b": BioMedGPTV,
    "kv-plm": KVPLM,
    "kv-plm*": KVPLM,
    "molfm": MolFM,
    "momu": MoMu,
    "text2mol": Text2MolMLP,
    "molt5": MolT5
}

SUPPORTED_TEXT_DECODER = {
    "molt5": MolT5,
    "chatmol": ChatMol,
    "biot5": BioT5,
}

SUPPORTED_KNOWLEDGE_ENCODER = {
    "TransE": TransE,
    "gin": GIN
}