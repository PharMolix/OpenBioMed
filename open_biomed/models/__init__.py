import logging
from open_biomed.models.agentic_models.llm_molecule_optimization import TextBasedMoleculeEditingAgent
from open_biomed.models.foundation_models.molt5 import MolT5
from open_biomed.models.foundation_models.biot5 import BioT5
from open_biomed.models.foundation_models.biot5_plus import BioT5_PLUS
from open_biomed.models.molecule.molcraft import MolCRAFT
from open_biomed.models.molecule.graphmvp import GraphMVP, GraphMVPRegression
from open_biomed.models.foundation_models.pharmolix_fm import PharmolixFM
from open_biomed.models.protein.mutaplm.mutaplm import MutaPLM
from open_biomed.models.task_models.protein_text_translation import EnsembleTextBasedProteinGenerationModel
from open_biomed.models.protein.esmfold.esmfold import EsmFold
from open_biomed.models.protein.codefp.codefp import CodeFP

MODEL_REGISTRY = {
    "text_based_molecule_editing": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS,
        "llm4molopt": TextBasedMoleculeEditingAgent,
    },
    "go_guided_protein_generation": {
        "codefp": CodeFP,
    },
    "structure_text_based_molecule_optimization": {
        "llm4molopt": TextBasedMoleculeEditingAgent,
    },
    "molecule_captioning": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "text_guided_molecule_generation": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "molecule_question_answering": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "text_based_protein_generation": {
        "biot5": BioT5,
        "ensemble": EnsembleTextBasedProteinGenerationModel,
    },
    "protein_question_answering": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "molecule_property_prediction": {
        "graphmvp": GraphMVP,
        "graphmvp_regression": GraphMVPRegression,
    },
    "pocket_molecule_docking": {
        "pharmolix_fm": PharmolixFM,
    },
    "structure_based_drug_design": {
        "pharmolix_fm": PharmolixFM,
        "molcraft": MolCRAFT,
    },
    "mutation_explanation": {
        "mutaplm": MutaPLM,
    },
    "mutation_engineering": {
        "mutaplm": MutaPLM,
    },
    "protein_folding": {
        "esmfold": EsmFold,
    },
    "go_guided_protein_generation": {
        "codefp": CodeFP,
    },
}

try:
    from open_biomed.models.cell.langcell.langcell import LangCell
    MODEL_REGISTRY["cell_annotation"] = {
        "langcell": LangCell,
    }
except ImportError:
    logging.warning("Install geneformer to use LangCell: pip install geneformer")