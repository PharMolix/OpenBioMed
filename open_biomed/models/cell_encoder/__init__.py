from models.cell_encoder.gat import *
from models.cell_encoder.performer import PerformerLM

SUPPORTED_CELL_ENCODER = {
    "scbert": PerformerLM
}