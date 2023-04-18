from models.drug_encoder.cnn import *
from models.drug_encoder.gin_tgsa import *
from models.drug_encoder.mgnn import *
from models.drug_encoder.momu import *
from models.drug_encoder.kv_plm import *
from models.drug_encoder.bert import *
from models.drug_encoder.biomedgpt import *
from models.drug_encoder.pyg_gnn import GraphMVP
from models.drug_encoder.molclr_gnn import *
from models.drug_encoder.text2mol import *

SUPPORTED_DRUG_ENCODER = {
    "cnn": CNN,
    "mgnn": MGNN,
    "kvplm": KVPLM,
    "graphcl": MoMuGNN,
    "molclr": GINet,
    "graphmvp": GraphMVP,
}