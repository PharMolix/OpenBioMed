from models.cell_encoder.gat import *
from models.cell_encoder.performer import PerformerLM
from models.cell_encoder.performer_celllm import PerformerLM_CellLM

SUPPORTED_CELL_ENCODER = {
    "scbert": PerformerLM,
    "celllm": PerformerLM_CellLM
}