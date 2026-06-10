from open_biomed.tools.tool_misc import *
from open_biomed.tools.web_request_tools import *
from open_biomed.tools.visualization_tools import *
from open_biomed.tools.third_party_tools import *
from open_biomed.data.molecule import *
from open_biomed.scripts.inference import *
from open_biomed.tools.drug_lead_analysis_tool import DrugLeadAnalysisTool
from open_biomed.tools.kegg_query_tool import KEGGQueryRequester
from open_biomed.tools.retrosynthesis_tool import RetrosynthesisRequester
from open_biomed.tools.disease_drug_intel_tool import DiseaseDrugIntelTool
from open_biomed.tools.drug_drug_interaction_tool import DrugDrugInteractionTool
from open_biomed.tools.literature_search_tool import LiteratureSearchTool
from open_biomed.tools.mutation_design_aav_tool import MutationDesignAAV
from open_biomed.tools.tfold_tool import TFoldRequester
from open_biomed.tools.iggm_tool import IgGMRequester
from open_biomed.tools.boltz2_tool import Boltz2Requester


from open_biomed.tools.file_reader_tools import ReadMoleculeFile, ReadProteinFile

# TODO: Add pocket prediction as a tool
class LazyDictForTool(dict):
    def available_tools(self):
        return [
            "text_based_molecule_editing", "molecule_property_prediction", "structure_based_drug_design",
            "molecule_question_answering", "protein_question_answering", "mutation_explanation",
            "mutation_engineering", "apply_mutation_to_sequence", "pocket_molecule_docking",
            "protein_molecule_docking_score", "protein_folding", "protein_binding_site_prediction",
            "visualize_molecule", "visualize_protein", "visualize_complex",
            "visualize_protein_pocket", "molecule_name_request", "pubchemid_search",
            "molecule_structure_request", "pubchem_bioactivity", "protein_uniprot_request", "protein_pdb_request",
            "ppi_string_request",
            "web_search", "import_pocket", "create_pocket_from_ligand", "export_molecule", "export_protein",
            "molecule_qed", "molecule_sa", "molecule_logp", "molecule_lipinski", "molecule_similarity",
            "molecule_property_calculation",
            "drug_lead_analysis",
            "extract_molecules_from_pdb_file", "analyze_complex_interaction", "summarize_content", "chembl_query",
            "kegg_query", "retrosynthesis", "disease_drug_intel", "ddi_analysis", "literature_search",
            "binding_affinity", "similar_protein_search",
            "mutation_design_aav",
            "tfold_antibody_structure",
            "iggm_antibody_design",
            "boltz2_structure_prediction",
            "read_molecule_file", "read_protein_file"
        ]
    
    def __missing__(self, key):
        if key == "text_based_molecule_editing":
            self[key] = test_text_based_molecule_editing(unit_test=False)
        elif key == "molecule_property_prediction":
            self[key] = test_molecule_property_prediction(unit_test=False)
        elif key == "structure_based_drug_design":
            self[key] = test_structure_based_drug_design(unit_test=False)
        elif key == "molecule_question_answering":
            self[key] = test_molecule_question_answering(unit_test=False)
        elif key == "protein_question_answering":
            self[key] = test_protein_question_answering(unit_test=False)
        elif key == "mutation_explanation":
            self[key] = test_mutation_explanation(unit_test=False)
        elif key == "mutation_engineering":
            self[key] = test_mutation_engineering(unit_test=False)
        elif key == "apply_mutation_to_sequence":
            self[key] = MutationToSequence()
        elif key == "pocket_molecule_docking":
            self[key] = test_pocket_molecule_docking(unit_test=False)
        elif key == "protein_molecule_docking_score":
            self[key] = test_protein_molecule_docking(unit_test=False)
        elif key == "protein_folding":
            self[key] = test_protein_folding(unit_test=False)
        elif key == "protein_binding_site_prediction":
            self[key] = ProteinBindingSitePrediction()
        elif key == "visualize_molecule":
            self[key] = PyMolVisualizerWrapper(task="visualize_molecule")
        elif key == "visualize_protein":
            self[key] = PyMolVisualizerWrapper(task="visualize_protein")
        elif key == "visualize_complex":
            self[key] = PyMolVisualizerWrapper(task="visualize_complex")
        elif key == "visualize_protein_pocket":
            self[key] = PyMolVisualizerWrapper(task="visualize_protein_pocket")
        # TODO: update the name mapping between frontend and backend
        elif key == "molecule_name_request" or key == "pubchemid_search":
            self[key] = PubChemRequester()
        elif key == "molecule_structure_request":
            self[key] = PubChemStructureRequester()
        elif key == "pubchem_bioactivity":
            self[key] = PubChemBioactivityRequester()
        elif key == "protein_uniprot_request":
            self[key] = UniProtRequester()
        elif key == "protein_pdb_request":
            self[key] = PDBRequester()
        elif key == "ppi_string_request":
            self[key] = STRINGRequester()
        elif key == "extract_molecules_from_pdb_file":
            self[key] = ExtractAllMoleculesFromPDB()
        elif key == "analyze_complex_interaction":
            self[key] = ComplexInteractionAnalysis()
        elif key == "web_search":
            self[key] = WebSearchRequester()
        # elif key == "key_info_extract":
        #    self[key] = KeyInfoExtractor()
        elif key == "import_pocket":
            self[key] = ImportPocket()
        elif key == "create_pocket_from_ligand":
            self[key] = CreatePocketFromLigand()
        elif key == "export_molecule":
            self[key] = ExportMolecule()
        elif key == "export_protein":
            self[key] = ExportProtein()
        elif key == "molecule_qed":
            self[key] = MoleculeQEDTool()  # TODO
        elif key == "molecule_sa":
            self[key] = MoleculeSATool()
        elif key == "molecule_logp":
            self[key] = MoleculeLogPTool()
        elif key == "molecule_lipinski":
            self[key] = MoleculeLipinskiTool()
        elif key == "molecule_property_calculation":
            self[key] = MoleculePropertyCalculationTool()
        elif key == "molecule_similarity":
            self[key] = MoleculeSimilarityTool()
        elif key == "drug_lead_analysis":
            self[key] = DrugLeadAnalysisTool()
        elif key == "summarize_content":
            self[key] = LLMSummarize()
        elif key == "chembl_query":
            self[key] = ChEMBLQueryRequester()
        elif key == "kegg_query":
            self[key] = KEGGQueryRequester()
        elif key == "retrosynthesis":
            self[key] = RetrosynthesisRequester()
        elif key == "disease_drug_intel":
            self[key] = DiseaseDrugIntelTool()
        elif key == "ddi_analysis":
            self[key] = DrugDrugInteractionTool()
        elif key == "literature_search":
            self[key] = LiteratureSearchTool()
        elif key == "binding_affinity":
            self[key] = ProdigyBindingAffinity()
        elif key == "similar_protein_search":
            self[key] = SimilarProteinSearch()
        elif key == "mutation_design_aav":
            self[key] = MutationDesignAAV()
        elif key == "tfold_antibody_structure":
            self[key] = TFoldRequester()
        elif key == "iggm_antibody_design":
            self[key] = IgGMRequester()
        elif key == "boltz2_structure_prediction":
            self[key] = Boltz2Requester()
        elif key == "read_molecule_file":
            self[key] = ReadMoleculeFile()
        elif key == "read_protein_file":
            self[key] = ReadProteinFile()
        else:
            raise NotImplementedError(f"{key} is currently not supported!")
        return self[key]

TOOLS = LazyDictForTool()