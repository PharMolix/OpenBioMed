from open_biomed.models.cell.gat import CellGAT
from open_biomed.models.cell.performer import PerformerLM
from open_biomed.models.cell.performer_celllm import PerformerLM_CellLM
from open_biomed.models.cell.deepcdr import DeepCDR
from open_biomed.models.cell.geneformer import GeneFormer

SUPPORTED_CELL_ENCODER = {
    "scbert": PerformerLM,
    "celllm": PerformerLM_CellLM,
    "gat": CellGAT,
    "deepcdr": DeepCDR,
    "geneformer": GeneFormer
}