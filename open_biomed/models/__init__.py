from models.molecule import *
from models.protein import *
from models.cell import *
from models.knowledge import *
from models.text import *
from models.multimodal import *
from models.multimodal.molkformer import *

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
    "molfm": MolFM,
    "molkformer":MolKFormer
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
    "celllm": PerformerLM_CellLM
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
}

SUPPORTED_KNOWLEDGE_ENCODER = {
    "TransE": TransE,
    "gin": GIN
}