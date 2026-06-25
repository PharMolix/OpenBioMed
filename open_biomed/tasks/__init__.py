from typing import Optional

from open_biomed.tasks.multi_modal_tasks.text_based_molecule_editing import TextMoleculeEditing
from open_biomed.tasks.multi_modal_tasks.molecule_text_translation import MoleculeCaptioning
from open_biomed.tasks.multi_modal_tasks.text_guided_molecule_generation import TextGuidedMoleculeGeneration
from open_biomed.tasks.multi_modal_tasks.molecule_question_answering import MoleculeQA
from open_biomed.tasks.multi_modal_tasks.protein_text_translation import TextBasedProteinGeneration
from open_biomed.tasks.multi_modal_tasks.protein_question_answering import ProteinQA
from open_biomed.tasks.multi_modal_tasks.mutation_text_translation import MutationExplanation, MutationEngineering
from open_biomed.tasks.multi_modal_tasks.go_guided_protein_generation import GoGuidedProteinGeneration
from open_biomed.tasks.aidd_tasks.molecule_property_prediction import MoleculePropertyPrediction, MoleculePropertyPredictionRegression
from open_biomed.tasks.aidd_tasks.protein_molecule_docking import PocketMoleculeDocking
from open_biomed.tasks.aidd_tasks.structure_based_drug_design import StructureBasedDrugDesign, StructureTextBasedMoleculeOptimization
from open_biomed.tasks.aidd_tasks.protein_folding import ProteinFolding
from open_biomed.tasks.aidd_tasks.cell_annotation import CellAnnotation
from open_biomed.tasks.aidd_tasks.mutation_design_aav import MutationDesignAAV
from open_biomed.tasks.aidd_tasks.mutation_design_gfp import MutationDesignGFP

TASK_REGISTRY = {
    "text_based_molecule_editing": TextMoleculeEditing,
    "molecule_captioning": MoleculeCaptioning,
    "text_guided_molecule_generation": TextGuidedMoleculeGeneration,
    "molecule_question_answering": MoleculeQA,
    "protein_question_answering": ProteinQA,
    "text_based_protein_generation": TextBasedProteinGeneration,
    "molecule_property_prediction": MoleculePropertyPrediction,
    "molecule_property_prediction_regression": MoleculePropertyPredictionRegression,
    "pocket_molecule_docking": PocketMoleculeDocking,
    "structure_based_drug_design": StructureBasedDrugDesign,
    "structure_text_based_molecule_optimization": StructureTextBasedMoleculeOptimization,
    "mutation_explanation": MutationExplanation,
    "mutation_engineering": MutationEngineering,
    "protein_folding": ProteinFolding,
    "cell_annotation": CellAnnotation,
    "go_guided_protein_generation": GoGuidedProteinGeneration,
    "mutation_design_aav": MutationDesignAAV,
    "mutation_design_gfp": MutationDesignGFP,
}
