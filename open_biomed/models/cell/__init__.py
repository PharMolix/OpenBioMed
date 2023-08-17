from models.cell.gat import CellGAT
from models.cell.performer import PerformerLM
from models.cell.performer_celllm import PerformerLM_CellLM
from models.cell.deepcdr import DeepCDR

SUPPORTED_CELL_ENCODER = {
    "scbert": PerformerLM,
    "celllm": PerformerLM_CellLM,
    "gat": CellGAT,
    "deepcdr": DeepCDR
}