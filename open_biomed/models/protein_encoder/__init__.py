from models.protein_encoder.cnn import CNN
from models.protein_encoder.mcnn import MCNN
from models.protein_encoder.prottrans import ProtTrans

SUPPORTED_PROTEIN_ENCODER = {
    "cnn": CNN,
    "mcnn": MCNN,
    "prottrans": ProtTrans
}