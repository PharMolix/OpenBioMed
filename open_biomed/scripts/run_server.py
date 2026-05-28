from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn.functional as F
import uvicorn
import random
import copy
import asyncio
import logging
import subprocess
import time
import uuid
from typing import Optional, List, Dict, Callable, Any, Literal

# import function
from open_biomed.data import Molecule, Text, Protein, Pocket
from open_biomed.tools.tool_misc import MutationToSequence
from open_biomed.core.oss_warpper import oss_warpper
from open_biomed.tools.tool_registry import TOOLS


app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('OpenBioMed')
logger.setLevel(logging.INFO)


class IO_Reader:
    def __init__(self):
        pass
    @staticmethod
    def get_molecule(string):
        if string.endswith(".sdf"):
            return Molecule.from_sdf_file(string)
        elif string.endswith(".pkl"):
            return Molecule.from_binary_file(string)
        else:
            return Molecule.from_smiles(string)
    
    @staticmethod
    def get_protein(string):
        if string.endswith(".pdb"):
            return Protein.from_pdb_file(string)
        elif string.endswith(".pkl"):
            return Protein.from_binary_file(string)
        else:
            return Protein.from_fasta(string)

    @staticmethod
    def get_pocket(string):
        return Pocket.from_binary_file(string)

    @staticmethod
    def get_text(string):
        return Text.from_str(string)




molecule_property_prediction_prompt = {
    "BBBP": "The blood-brain barrier prediction result is {output}. "
            "This result indicates the model's prediction of whether the compound can effectively penetrate the blood-brain barrier. "
            "A positive result suggests that the compound may have potential for central nervous system targeting, "
            "while a negative result implies limited permeability.",
    
    "ClinTox": "The clinical toxicity prediction result is {output}. "
                "This result reflects the model's assessment of the likelihood that the compound will fail clinical trials due to toxicity concerns. "
                "A positive result indicates a higher risk of toxicity, while a negative result suggests the compound is less likely to exhibit significant toxicity in clinical settings.",
    
    "Tox21": "The Tox21 toxicity assessment result is {output}. "
             "This result provides an evaluation of the compound's potential toxicity, focusing on nuclear receptors and stress response pathways. "
             "A positive result indicates the presence of toxic effects, while a negative result suggests the compound is less likely to exhibit these toxicities.",
    
    "ToxCast": "The ToxCast toxicity screening result is {output}. "
               "This result is based on high-throughput in vitro screening and indicates the compound's potential toxicity profile. "
               "A positive result suggests significant toxicity, while a negative result implies lower toxicity risk.",
    
    "SIDER": "The SIDER adverse drug reaction analysis result is {output}. "
             "This result provides insights into the potential adverse drug reactions (ADRs) associated with the compound. "
             "A positive result indicates a higher likelihood of adverse reactions, while a negative result suggests fewer potential ADRs.",
    
    "HIV": "The HIV inhibition prediction result is {output}. "
           "This result indicates the model's prediction of the compound's ability to inhibit HIV replication. "
           "A positive result suggests strong inhibitory activity, while a negative result implies limited effectiveness against HIV.",
    
    "BACE": "The BACE-1 activity prediction result is {output}. "
            "This result provides a prediction of the compound's binding affinity to human β-secretase 1 (BACE-1). "
            "A positive result indicates strong binding activity, suggesting potential as a BACE-1 inhibitor, while a negative result implies weaker binding.",
    
    "MUV": "The MUV virtual screening validation result is {output}. "
           "This result indicates the model's assessment of the compound's potential as a hit in virtual screening. "
           "A positive result suggests the compound is likely to be active against the target, while a negative result implies lower activity."
}


# Define the request body model
class TaskRequest(BaseModel):
    task: str
    model: Optional[str] = None
    config: Optional[str] = None
    visualize: Optional[str] = None
    molecule: Optional[str] = None
    protein: Optional[str] = None
    pocket: Optional[str] = None
    text: Optional[str] = None
    dataset: Optional[str] = None
    query: Optional[str] = None
    mutation: Optional[str] = None
    indices: Optional[str] = None
    property: Optional[Literal["QED", "SA", "LogP", "Lipinski"]] = None
    molecule_1: Optional[str] = None
    molecule_2: Optional[str] = None
    similarity: Optional[float] = None
    value: Optional[str] = None
    # ChEMBL query extension
    query_type: Optional[str] = None
    target_name: Optional[str] = None
    uniprot_id: Optional[str] = None
    molecule_name: Optional[str] = None
    chembl_id: Optional[str] = None
    disease: Optional[str] = None
    standard_type: Optional[str] = None
    standard_value_lte: Optional[int] = None
    max_phase: Optional[int] = None
    limit: Optional[int] = None
    # KEGG query extension
    database: Optional[str] = None
    option: Optional[str] = None
    entry_id: Optional[str] = None
    target_db: Optional[str] = None
    source_id: Optional[str] = None
    # PPI STRING query extension
    species: Optional[int] = None
    required_score: Optional[int] = None
    # ClinicalTrials query extension
    query_cond: Optional[str] = None
    query_term: Optional[str] = None
    filter_overall_status: Optional[str] = None
    fields: Optional[List[str]] = None
    sort: Optional[List[str]] = None
    page_size: Optional[int] = None
    count_total: Optional[bool] = None
    page_token: Optional[str] = None
    nct_id: Optional[str] = None
    # Tavily Search extension
    max_results: Optional[int] = None
    api_key: Optional[str] = None
    # ChEMBL additional fields
    molecule_chembl_id: Optional[str] = None
    efo_term: Optional[str] = None
    offset: Optional[int] = None
    # DDI analysis fields
    drugs: Optional[List[str]] = None
    drug_ids: Optional[str] = None
    drug_id: Optional[str] = None
    # Literature search fields
    pmids: Optional[List[str]] = None
    max_results: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[int] = None
    category: Optional[str] = None
    # PDB request mode
    mode: Optional[str] = None  # "metadata" or "file_only"
    # PubChem bioactivity query fields
    cid: Optional[int] = None
    aid: Optional[int] = None
    aids_type: Optional[str] = None  # "active" or "inactive"
    cids_type: Optional[str] = None  # "active" or "inactive"
    gene_symbol: Optional[str] = None
    gene_id: Optional[int] = None
    max_records: Optional[int] = None
    # Binding affinity prediction fields
    protein_complex: Optional[str] = None  # PDB file path for protein complex
    distance_cutoff: Optional[float] = None  # Distance cutoff for PRODIGY


class SearchRequest(BaseModel):
    task: str
    query: Optional[str] = None
    molecule: Optional[str] = None
    threshold: Optional[str] = None
    mode: Optional[str] = None  # For PDBRequester: "metadata" or "file_only"


class TaskConfig:
    def __init__(self, task_name: str, required_inputs: List[str], pipeline_key: str, handler_function: Callable, is_async: bool = False):
        self.task_name = task_name
        self.required_inputs = required_inputs
        self.pipeline_key = pipeline_key
        self.handler_function = handler_function
        self.is_async = is_async

    def validate_inputs(self, request: Dict[str, Any]):
        missing_inputs = [key for key in self.required_inputs if key not in request]
        if missing_inputs:
            raise HTTPException(status_code=400, detail=f"Missing required inputs: {', '.join(missing_inputs)}")



class TaskLoader:
    def __init__(self):
        self.tasks = {}

    def register_task(self, task_config: TaskConfig):
        self.tasks[task_config.task_name] = task_config

    def get_task(self, task_name: str):
        task = self.tasks.get(task_name)
        if not task:
            raise HTTPException(status_code=400, detail="Invalid task_name")
        return task



# Handlers for run_pipeline
def handle_text_based_molecule_editing(request: TaskRequest, pipeline):
    required_inputs = ["molecule", "text"]
    molecule = IO_Reader.get_molecule(request.molecule)
    text = IO_Reader.get_text(request.text)
    outputs = pipeline.run(molecule=molecule, text=text)
    smiles = outputs[0][0].smiles
    path = outputs[1][0]
    return {"task": request.task, "model": request.model, "molecule": path, "molecule_preview": smiles}

def handle_structure_based_drug_design(request: TaskRequest, pipeline):
    required_inputs = ["pocket"]
    pocket = Pocket.from_binary_file(request.pocket)
    outputs = pipeline.run(pocket=pocket)
    smiles = outputs[0][0].smiles
    path = outputs[1][0]
    return {"task": request.task, "model": request.model, "molecule": path, "molecule_preview": smiles}

def handle_molecule_question_answering(request: TaskRequest, pipeline):
    required_inputs = ["text", "molecule"]
    text = IO_Reader.get_text(request.text)
    molecule = IO_Reader.get_molecule(request.molecule)
    outputs = pipeline.run(molecule=molecule, text=text)
    text = outputs[0][0].str
    return {"task": request.task, "model": request.model, "text": text}

def handle_protein_question_answering(request: TaskRequest, pipeline):
    required_inputs = ["text", "protein"]
    text = IO_Reader.get_text(request.text)
    protein = IO_Reader.get_protein(request.protein)
    outputs = pipeline.run(protein=protein, text=text)
    text = outputs[0][0].str
    return {"task": request.task, "model": request.model, "text": text}

def handle_visualize_molecule(request: TaskRequest, pipeline):
    required_inputs = ["molecule"]
    #ligand = Molecule.from_binary_file(request.molecule)
    #outputs = pipeline.run(ligand, config="ball_and_stick", rotate=False)
    vis_process = [
                    "python3", "./open_biomed/tools/visualization_tools.py",
                    "--task", "visualize_molecule",
                    "--molecule_config", request.visualize,
                    "--save_output_filename", "./tmp/molecule_visualization_file.txt",
                    "--molecule", request.molecule]
    subprocess.Popen(vis_process).communicate()
    outputs = open("./tmp/molecule_visualization_file.txt", "r").read()
    oss_file_path = oss_warpper.generate_file_name(outputs)
    outputs = oss_warpper.upload(oss_file_path, outputs)
    return {"task": request.task, "image": outputs}

def handle_visualize_complex(request: TaskRequest, pipeline):
    required_inputs = ["protein", "molecule"]
    #ligand = Molecule.from_binary_file(request.molecule)
    #protein = Protein.from_pdb_file(request.protein)
    #outputs = pipeline.run(molecule=ligand, protein=protein, rotate=True)
    vis_process = [
                    "python3", "./open_biomed/tools/visualization_tools.py",
                    "--task", "visualize_complex",
                    "--save_output_filename", "./tmp/complex_visualization_file.txt",
                    "--molecule", request.molecule,
                    "--protein", request.protein]
    subprocess.Popen(vis_process).communicate()
    outputs = open("./tmp/complex_visualization_file.txt", "r").read()
    oss_file_path = oss_warpper.generate_file_name(outputs)
    outputs = oss_warpper.upload(oss_file_path, outputs)
    return {"task": request.task, "image": outputs}

def handle_molecule_property_prediction(request: TaskRequest, pipeline):
    required_inputs = ["molecule", "dataset"]
    molecule = IO_Reader.get_molecule(request.molecule)
    dataset = IO_Reader.get_text(request.dataset)
    outputs = pipeline.run(molecule=molecule, task=dataset.str)

    #output = outputs[0][0].cpu()
    #output = F.softmax(output, dim=0).tolist()
    return {"task": request.task, "model": request.model, "score": outputs[0][0]}

def handle_protein_binding_site_prediction(request: TaskRequest, pipeline):
    required_inputs = ["protein"]
    protein = IO_Reader.get_protein(request.protein)
    outputs = pipeline.run(protein=protein)
    output = outputs[1][0]
    pocket_preview = str(outputs[0][0])
    return {"task": request.task, "model": request.model, "pocket": output, "pocket_preview": pocket_preview}

def handle_protein_folding(request: TaskRequest, pipeline):
    required_inputs = ["protein"]
    protein = IO_Reader.get_protein(request.protein)
    outputs = pipeline.run(protein=protein)
    protein = outputs[1][0]
    return {"task": request.task, "model": request.model, "protein": protein}


# Handlers for web_search
async def handle_molecule_name_request(request: SearchRequest, requester):
    outputs = await requester.run_async(request.query)
    smiles = outputs[0][0].smiles
    output = outputs[1][0]
    return {"task": request.task, "molecule": output, "molecule_preview": smiles}

async def handle_web_search(request: SearchRequest, requester):
    outputs = await requester.run_async(request.query)
    outputs = outputs[0][0]
    return {"task": request.task, "text": outputs}

async def handle_molecule_structure_request(request: SearchRequest, requester):
    molecule = IO_Reader.get_molecule(request.molecule)
    threshold = request.threshold
    outputs = await requester.run_async(molecule, threshold=float(threshold), max_records=1)
    # Pick a random molecule
    index = random.randint(0, len(outputs[1])-1)
    molecule = outputs[1][index]
    molecule_preview = outputs[0][index].smiles
    return {"task": request.task, "molecule": molecule, "molecule_preview": molecule_preview}


async def handle_pubchem_bioactivity(request: TaskRequest, requester):
    """Handle PubChem bioactivity queries."""
    kwargs = {
        "cid": request.cid,
        "aid": request.aid,
        "aids_type": request.aids_type,
        "cids_type": request.cids_type,
        "gene_symbol": request.gene_symbol,
        "gene_id": request.gene_id,
        "max_records": request.max_records or 10
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


async def handle_protein_uniprot_request(request: SearchRequest, requester):
    outputs = await requester.run_async(request.query)
    outputs = outputs[1][0]
    protein = IO_Reader.get_protein(outputs)
    protein_preview = str(protein)
    return {"task": request.task, "protein": outputs, "protein_preview": protein_preview}


async def handle_protein_pdb_request(request: TaskRequest, requester):
    mode = request.mode or "file_only"  # Default to file_only for drug discovery workflows
    outputs = await requester.run_async(request.query, mode=mode)
    if mode == "file_only":
        # outputs[0] is the file path list, outputs[1] is the content list
        pdb_file = outputs[0][0]
        protein = IO_Reader.get_protein(pdb_file)
        protein_preview = str(protein)
        return {"task": request.task, "protein": pdb_file, "protein_preview": protein_preview}
    else:
        # metadata mode returns JSON content
        outputs = outputs[1][0]
        return {"task": request.task, "protein": outputs}


def handle_mutation_explanation(request: TaskRequest, pipeline):
    required_inputs = ["protein", "mutation"]
    mutation = request.mutation
    protein = IO_Reader.get_protein(request.protein)
    outputs = pipeline.run(protein=protein, mutation=mutation)
    output = outputs[0][0]
    return {"task": request.task, "model":request.model, "text": output}


def handle_mutation_engineering(request: TaskRequest, pipeline):
    required_inputs = ["protein", "text"]
    protein = IO_Reader.get_protein(request.protein)
    text = IO_Reader.get_text(request.text)
    outputs = pipeline.run(protein=protein, text=text)
    mutation_list = copy.deepcopy(outputs[0][0][:50])
    mutation = random.choice(outputs[0][0])
    converter = MutationToSequence()
    outputs = converter.run([protein], [mutation])
    protein =  outputs[1][0]
    protein_preview = outputs[0][0].sequence
    return {"task": request.task, "model":request.model, "mutation": mutation_list, "protein": protein, "protein_preview": protein_preview}


def handle_pocket_molecule_docking(request: TaskRequest, pipeline):
    required_inputs = ["pocket", "molecule"]
    pocket = Pocket.from_binary_file(request.pocket)
    molecule = IO_Reader.get_molecule(request.molecule)
    outputs = pipeline.run(pocket=pocket, molecule=molecule)
    output = outputs[1][0]
    return {"task": request.task, "model":request.model, "molecule": output}


def handle_protein_molecule_docking_score(request: TaskRequest, pipeline):
    required_inputs = ["protein", "molecule"]
    protein = IO_Reader.get_protein(request.protein)
    molecule = IO_Reader.get_molecule(request.molecule)
    outputs = pipeline.run(protein=protein, molecule=molecule)
    output = outputs[0][0]
    return {"task": request.task, "model":request.model, "score": str(output)}


def handle_visualize_protein(request: TaskRequest, pipeline):
    required_inputs = ["protein"]
    protein = IO_Reader.get_protein(request.protein)
    vis_process = [
                    "python3", "./open_biomed/tools/visualization_tools.py",
                    "--task", "visualize_protein",
                    "--protein_config", request.visualize,
                    "--save_output_filename", "./tmp/protein_visualization_file.txt",
                    "--protein", request.protein]
    subprocess.Popen(vis_process).communicate()
    outputs = open("./tmp/protein_visualization_file.txt", "r").read()
    oss_file_path = oss_warpper.generate_file_name(outputs)
    outputs = oss_warpper.upload(oss_file_path, outputs)
    return {"task": request.task, "image": outputs}


def handle_visualize_protein_pocket(request: TaskRequest, pipeline):
    required_inputs = ["protein", "pocket"]
    protein = IO_Reader.get_protein(request.protein)
    pocket = IO_Reader.get_pocket(request.pocket)
    vis_process = [
                    "python3", "./open_biomed/tools/visualization_tools.py",
                    "--task", "visualize_protein_pocket",
                    "--save_output_filename", "./tmp/protein_pocket_visualization_file.txt",
                    "--protein", request.protein,
                    "--pocket", request.pocket]
    subprocess.Popen(vis_process).communicate()
    outputs = open("./tmp/protein_pocket_visualization_file.txt", "r").read()
    oss_file_path = oss_warpper.generate_file_name(outputs)
    outputs = oss_warpper.upload(oss_file_path, outputs)
    return {"task": request.task, "image": outputs}


def handle_export_molecule(request: TaskRequest, pipeline):
    required_inputs = ["molecule"]
    molecule = IO_Reader.get_molecule(request.molecule)
    files = pipeline.run([molecule])
    oss_file_path = oss_warpper.generate_file_name(files[0])
    outputs = oss_warpper.upload(oss_file_path, files[0])
    return {"task": request.task, "file": outputs}


def handle_export_protein(request: TaskRequest, pipeline):
    required_inputs = ["protein"]
    protein = IO_Reader.get_protein(request.protein)
    files = pipeline.run([protein])
    oss_file_path = oss_warpper.generate_file_name(files[0])
    outputs = oss_warpper.upload(oss_file_path, files[0])
    return {"task": request.task, "file": outputs}


def handle_import_pocket(request: TaskRequest, pipeline):
    required_inputs = ["protein", "indices"]
    protein = IO_Reader.get_protein(request.protein)
    indices = [int(i) - 1 for i in request.indices.split(";")]
    pockets, files = pipeline.run([protein], [indices])
    return {"task": request.task, "pocket": files[0], "pocket_preview": str(pockets[0])}


def handle_create_pocket_from_ligand(request: TaskRequest, pipeline):
    """Create a binding pocket from protein and reference ligand coordinates."""
    required_inputs = ["protein", "molecule"]
    protein = IO_Reader.get_protein(request.protein)
    ligand = IO_Reader.get_molecule(request.molecule)
    radius = request.similarity if request.similarity is not None else 10.0  # Use similarity field as radius
    pockets, files = pipeline.run(protein=protein, ligand=ligand, radius=radius)
    return {"task": request.task, "pocket": files[0], "pocket_preview": str(pockets[0])}


def handle_analyze_complex_interaction(request: TaskRequest, pipeline):
    """Analyze interactions between a molecule and protein."""
    required_inputs = ["molecule", "protein"]
    molecule = IO_Reader.get_molecule(request.molecule)
    protein = IO_Reader.get_protein(request.protein)
    reports, _ = pipeline.run(molecule=[molecule], protein=[protein])
    return {"task": request.task, "report": reports[0]}


def handle_binding_affinity(request: TaskRequest, pipeline):
    """
    Predict binding affinity for protein-protein complexes using PRODIGY.

    Inputs:
        - protein_complex: PDB file path containing the protein complex
        - distance_cutoff: (optional) Distance cutoff for calculating ICs, default 5.5

    Outputs:
        - binding_affinity: Predicted binding affinity score (kcal.mol-1)
        - description: Description message
    """
    protein_complex = request.protein_complex
    distance_cutoff = request.distance_cutoff or 5.5

    if not protein_complex:
        raise ValueError("protein_complex is required for binding affinity prediction")

    outputs, messages = pipeline.run(
        protein_complex=protein_complex,
        distance_cutoff=distance_cutoff
    )

    binding_affinity = outputs[0]
    description = messages[0]

    return {
        "task": request.task,
        "binding_affinity": binding_affinity,
        "distance_cutoff": distance_cutoff,
        "description": description
    }


# 25
def handle_molecule_similarity(request: TaskRequest, pipeline):
    required_inputs = ["molecule_1", "molecule_2"]
    molecule_1 = IO_Reader.get_molecule(request.molecule_1)
    molecule_2 = IO_Reader.get_molecule(request.molecule_2)
    outputs = pipeline.run(molecule_1=molecule_1, molecule_2=molecule_2)
    return {"task": request.task, "model":request.model, "similarity": outputs}

# 26
def handle_molecule_property_calculation(request: TaskRequest, pipeline):
    required_inputs = ["molecule", "property"]
    molecule = IO_Reader.get_molecule(request.molecule)
    property = request.property
    outputs = pipeline.run(molecule=molecule, property=property)
    return {"task": request.task, "model":request.model, "score": round(outputs, 5)}

def handle_drug_lead_analysis(request: TaskRequest, pipeline):
    required_inputs = ["molecule"]
    molecule = IO_Reader.get_molecule(request.molecule)
    outputs, messages = pipeline.run(molecule=molecule)
    return {"task": request.task, "model": request.model, "report": outputs[0]}


async def handle_chembl_query(request: TaskRequest, requester):
    kwargs = request.model_dump(exclude={"task", "query_type"}, exclude_none=True)
    # Remove non-ChEMBL fields that may have been populated
    for key in ["model", "config", "visualize", "molecule", "protein", "pocket",
                "text", "dataset", "query", "mutation", "indices", "property",
                "molecule_1", "molecule_2", "similarity", "value",
                "database", "option", "entry_id", "target_db", "source_id",
                "species", "required_score"]:
        kwargs.pop(key, None)
    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


async def handle_ppi_string_request(request: TaskRequest, requester):
    results, _ = await requester.run_async(
        uniprot_id=request.uniprot_id,
        species=request.species or 9606,
        required_score=request.required_score or 700,
        limit=request.limit or 50
    )
    return {"task": request.task, "uniprot_id": request.uniprot_id, "results": results}


async def handle_kegg_query(request: TaskRequest, requester):
    kwargs = request.model_dump(exclude={"task", "query_type"}, exclude_none=True)
    # Remove non-KEGG fields
    for key in ["model", "config", "visualize", "molecule", "protein", "pocket",
                "text", "dataset", "mutation", "indices", "property",
                "molecule_1", "molecule_2", "similarity", "value",
                "target_name", "uniprot_id", "molecule_name", "chembl_id",
                "disease", "standard_type", "standard_value_lte", "max_phase", "limit",
                "species", "required_score"]:
        kwargs.pop(key, None)
    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


async def handle_retrosynthesis(request: TaskRequest, requester):
    kwargs = request.model_dump(exclude={"task", "query_type"}, exclude_none=True)
    for key in ["model", "config", "visualize", "protein", "pocket",
                "text", "dataset", "mutation", "indices", "property",
                "molecule_1", "molecule_2", "similarity", "value",
                "target_name", "uniprot_id", "molecule_name", "chembl_id",
                "disease", "standard_type", "standard_value_lte", "max_phase", "limit",
                "species", "required_score", "database", "option", "entry_id",
                "target_db", "source_id"]:
        kwargs.pop(key, None)
    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


async def handle_disease_drug_intel(request: TaskRequest, requester):
    """Handle disease-drug intelligence queries."""
    kwargs = request.model_dump(exclude={"task", "query_type"}, exclude_none=True)
    # Remove non-relevant fields for this task
    for key in ["model", "config", "visualize", "protein", "pocket",
                "text", "dataset", "mutation", "indices", "property",
                "molecule_1", "molecule_2", "similarity", "value",
                "uniprot_id", "molecule_name",
                "species", "required_score",
                "database", "option", "entry_id", "target_db", "source_id",
                "standard_type", "standard_value_lte", "max_phase",
                "drug_ids", "drug_id"]:
        kwargs.pop(key, None)

    # Map target_name to query for ChEMBL target search
    if request.query_type == "chembl_search_target" and "target_name" in kwargs:
        kwargs["query"] = kwargs.pop("target_name")

    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


async def handle_ddi_analysis(request: TaskRequest, requester):
    """Handle drug-drug interaction analysis queries."""
    kwargs = request.model_dump(exclude={"task", "query_type"}, exclude_none=True)
    # Remove non-relevant fields for this task
    for key in ["model", "config", "visualize", "protein", "pocket",
                "text", "dataset", "mutation", "indices", "property",
                "molecule_1", "molecule_2", "similarity", "value",
                "target_name", "uniprot_id", "molecule_name", "chembl_id",
                "species", "required_score",
                "database", "option", "entry_id", "target_db", "source_id",
                "standard_type", "standard_value_lte", "max_phase",
                "query_cond", "query_term", "filter_overall_status",
                "fields", "sort", "page_size", "count_total", "page_token",
                "nct_id", "molecule_chembl_id", "efo_term", "offset",
                "disease", "limit",
                "pmids", "max_results", "start_date", "end_date", "days", "category"]:
        kwargs.pop(key, None)

    # Handle drugs parameter - can be list or comma-separated string
    if "drugs" in kwargs and isinstance(kwargs["drugs"], str):
        kwargs["drugs"] = kwargs["drugs"].split(",")

    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


async def handle_literature_search(request: TaskRequest, requester):
    """Handle biomedical literature search queries."""
    kwargs = request.model_dump(exclude={"task", "query_type"}, exclude_none=True)
    # Remove non-relevant fields for this task
    for key in ["model", "config", "visualize", "protein", "pocket",
                "text", "dataset", "mutation", "indices", "property",
                "molecule_1", "molecule_2", "similarity", "value",
                "target_name", "uniprot_id", "molecule_name", "chembl_id",
                "species", "required_score",
                "database", "option", "entry_id", "target_db", "source_id",
                "standard_type", "standard_value_lte", "max_phase",
                "filter_overall_status",
                "fields", "sort", "page_size", "count_total", "page_token",
                "nct_id", "molecule_chembl_id", "efo_term", "offset",
                "disease", "limit",
                "drugs", "drug_ids", "drug_id",
                "query_cond", "query_term"]:
        kwargs.pop(key, None)

    # Handle pmids parameter - can be list or comma-separated string
    if "pmids" in kwargs and isinstance(kwargs["pmids"], str):
        kwargs["pmids"] = kwargs["pmids"].split(",")

    results, _ = await requester.run_async(request.query_type, **kwargs)
    return {"task": request.task, "query_type": request.query_type, "results": results}


def handle_extract_molecules_from_pdb_file(request: TaskRequest, pipeline):
    """Extract proteins, ligands, and ions from a PDB file."""
    required_inputs = ["protein"]
    # protein field contains the PDB file path
    pdb_file = request.protein
    outputs_list, metadata_list = pipeline.run(pdb_file=pdb_file)

    # The serial_exec wrapper returns [output], [msg]
    # outputs_list = [results] where results is a list of tuples (type, chain_id, obj)
    outputs = outputs_list[0]
    metadata = metadata_list[0]

    # Prepare results with file paths
    results = []
    for item_type, chain_id, obj in outputs:
        if item_type == "protein":
            # Save protein to file
            protein_file = obj.save_binary()
            results.append({
                "type": "protein",
                "chain_id": chain_id,
                "name": obj.name,
                "sequence_preview": str(obj)[:100],
                "file": protein_file
            })
        elif item_type == "molecule":
            # Save molecule to file
            mol_file = obj.save_binary()
            results.append({
                "type": "molecule",
                "chain_id": chain_id,
                "name": obj.name,
                "smiles": obj.smiles,
                "file": mol_file
            })
        elif item_type == "ion":
            results.append({
                "type": "ion",
                "chain_id": chain_id,
                "name": obj.name if hasattr(obj, 'name') else "unknown_ion"
            })

    return {"task": request.task, "results": results, "metadata": metadata}


TASK_CONFIGS = [
    {
        "task_name": "text_based_molecule_editing",
        "required_inputs": ["molecule", "text"],
        "pipeline_key": "text_based_molecule_editing",
        "handler_function": handle_text_based_molecule_editing,
        "is_async": False
    },
    {
        "task_name": "structure_based_drug_design",
        "required_inputs": ["pocket"],
        "pipeline_key": "structure_based_drug_design",
        "handler_function": handle_structure_based_drug_design,
        "is_async": False
    },
    {
        "task_name": "molecule_question_answering",
        "required_inputs": ["text", "molecule"],
        "pipeline_key": "molecule_question_answering",
        "handler_function": handle_molecule_question_answering,
        "is_async": False
    },
    {
        "task_name": "protein_question_answering",
        "required_inputs": ["text", "protein"],
        "pipeline_key": "protein_question_answering",
        "handler_function": handle_protein_question_answering,
        "is_async": False
    },
    {
        "task_name": "visualize_molecule",
        "required_inputs": ["visualize", "molecule"],
        "pipeline_key": "visualize_molecule",
        "handler_function": handle_visualize_molecule,
        "is_async": False
    },
    {
        "task_name": "visualize_complex",
        "required_inputs": ["protein", "molecule"],
        "pipeline_key": "visualize_complex",
        "handler_function": handle_visualize_complex,
        "is_async": False
    },
    {
        "task_name": "visualize_protein",
        "required_inputs": ["visualize", "protein"],
        "pipeline_key": "visualize_protein",
        "handler_function": handle_visualize_protein,
        "is_async": False
    },
    {
        "task_name": "visualize_protein_pocket",
        "required_inputs": ["protein", "pocket"],
        "pipeline_key": "visualize_protein_pocket",
        "handler_function": handle_visualize_protein_pocket,
        "is_async": False
    },
    {
        "task_name": "molecule_property_prediction",
        "required_inputs": ["molecule", "dataset"],
        "pipeline_key": "molecule_property_prediction",
        "handler_function": handle_molecule_property_prediction,
        "is_async": False
    },
    {
        "task_name": "protein_binding_site_prediction",
        "required_inputs": ["protein"],
        "pipeline_key": "protein_binding_site_prediction",
        "handler_function": handle_protein_binding_site_prediction,
        "is_async": False
    },
    {
        "task_name": "protein_folding",
        "required_inputs": ["protein"],
        "pipeline_key": "protein_folding",
        "handler_function": handle_protein_folding,
        "is_async": False
    },
    {
        "task_name": "molecule_name_request",
        "required_inputs": ["query"],
        "pipeline_key": "molecule_name_request",
        "handler_function": handle_molecule_name_request,
        "is_async": True
    },
    {
        "task_name": "web_search",
        "required_inputs": ["query"],
        "pipeline_key": "web_search",
        "handler_function": handle_web_search,
        "is_async": True
    },
    {
        "task_name": "molecule_structure_request",
        "required_inputs": ["molecule", "threshold"],
        "pipeline_key": "molecule_structure_request",
        "handler_function": handle_molecule_structure_request,
        "is_async": True
    },
    {
        "task_name": "pubchem_bioactivity",
        "required_inputs": ["query_type"],
        "pipeline_key": "pubchem_bioactivity",
        "handler_function": handle_pubchem_bioactivity,
        "is_async": True
    },
    {
        "task_name": "protein_uniprot_request",
        "required_inputs": ["query"],
        "pipeline_key": "protein_uniprot_request",
        "handler_function": handle_protein_uniprot_request,
        "is_async": True
    },
    {
        "task_name": "protein_pdb_request",
        "required_inputs": ["query"],
        "pipeline_key": "protein_pdb_request",
        "handler_function": handle_protein_pdb_request,
        "is_async": True
    },
    {
        "task_name": "mutation_explanation",
        "required_inputs": ["mutation", "protein"],
        "pipeline_key": "mutation_explanation",
        "handler_function": handle_mutation_explanation,
        "is_async": False
    },
    {
        "task_name": "mutation_engineering",
        "required_inputs": ["text", "protein"],
        "pipeline_key": "mutation_engineering",
        "handler_function": handle_mutation_engineering,
        "is_async": False
    },
    {
        "task_name": "pocket_molecule_docking",
        "required_inputs": ["pocket", "molecule"],
        "pipeline_key": "pocket_molecule_docking",
        "handler_function": handle_pocket_molecule_docking,
        "is_async": False
    },
    {
        "task_name": "protein_molecule_docking_score",
        "required_inputs": ["protein", "molecule"],
        "pipeline_key": "protein_molecule_docking_score",
        "handler_function": handle_protein_molecule_docking_score,
        "is_async": False
    },
    {
        "task_name": "export_molecule",
        "required_inputs": ["molecule"],
        "pipeline_key": "export_molecule",
        "handler_function": handle_export_molecule,
        "is_async": False
    },
    {
        "task_name": "export_protein",
        "required_inputs": ["protein"],
        "pipeline_key": "export_protein",
        "handler_function": handle_export_protein,
        "is_async": False
    },
    {
        "task_name": "import_pocket",
        "required_inputs": ["pocket", "indices"],
        "pipeline_key": "import_pocket",
        "handler_function": handle_import_pocket,
        "is_async": False
    },
    {
        "task_name": "molecule_similarity",
        "required_inputs": ["molecule_1", "molecule_2"],
        "pipeline_key": "molecule_similarity",
        "handler_function": handle_molecule_similarity,
        "is_async": False
    },
    {
        "task_name": "molecule_property_calculation",
        "required_inputs": ["molecule", "property"],
        "pipeline_key": "molecule_property_calculation",
        "handler_function": handle_molecule_property_calculation,
        "is_async": False
    },
    {
        "task_name": "drug_lead_analysis",
        "required_inputs": ["molecule"],
        "pipeline_key": "drug_lead_analysis",
        "handler_function": handle_drug_lead_analysis,
        "is_async": False
    },
    {
        "task_name": "chembl_query",
        "required_inputs": ["query_type"],
        "pipeline_key": "chembl_query",
        "handler_function": handle_chembl_query,
        "is_async": True
    },
    {
        "task_name": "kegg_query",
        "required_inputs": ["query_type"],
        "pipeline_key": "kegg_query",
        "handler_function": handle_kegg_query,
        "is_async": True
    },
    {
        "task_name": "ppi_string_request",
        "required_inputs": ["uniprot_id"],
        "pipeline_key": "ppi_string_request",
        "handler_function": handle_ppi_string_request,
        "is_async": True
    },
    {
        "task_name": "retrosynthesis",
        "required_inputs": ["query_type"],
        "pipeline_key": "retrosynthesis",
        "handler_function": handle_retrosynthesis,
        "is_async": True
    },
    {
        "task_name": "disease_drug_intel",
        "required_inputs": ["query_type"],
        "pipeline_key": "disease_drug_intel",
        "handler_function": handle_disease_drug_intel,
        "is_async": True
    },
    {
        "task_name": "ddi_analysis",
        "required_inputs": ["query_type"],
        "pipeline_key": "ddi_analysis",
        "handler_function": handle_ddi_analysis,
        "is_async": True
    },
    {
        "task_name": "literature_search",
        "required_inputs": ["query_type"],
        "pipeline_key": "literature_search",
        "handler_function": handle_literature_search,
        "is_async": True
    },
    {
        "task_name": "extract_molecules_from_pdb_file",
        "required_inputs": ["protein"],
        "pipeline_key": "extract_molecules_from_pdb_file",
        "handler_function": handle_extract_molecules_from_pdb_file,
        "is_async": False
    },
    {
        "task_name": "create_pocket_from_ligand",
        "required_inputs": ["protein", "molecule"],
        "pipeline_key": "create_pocket_from_ligand",
        "handler_function": handle_create_pocket_from_ligand,
        "is_async": False
    },
    {
        "task_name": "analyze_complex_interaction",
        "required_inputs": ["molecule", "protein"],
        "pipeline_key": "analyze_complex_interaction",
        "handler_function": handle_analyze_complex_interaction,
        "is_async": False
    },
    {
        "task_name": "binding_affinity",
        "required_inputs": ["protein_complex"],
        "pipeline_key": "binding_affinity",
        "handler_function": handle_binding_affinity,
        "is_async": False
    }


]


task_loader = TaskLoader()

for task_config in TASK_CONFIGS:
    task_loader.register_task(TaskConfig(
        task_name=task_config["task_name"],
        required_inputs=task_config["required_inputs"],
        pipeline_key=task_config["pipeline_key"],
        handler_function=task_config["handler_function"],
        is_async=task_config["is_async"]
    ))

for task_config in TASK_CONFIGS:
    task_loader.register_task(TaskConfig(
        task_name=task_config["task_name"],
        required_inputs=task_config["required_inputs"],
        pipeline_key=task_config["pipeline_key"],
        handler_function=task_config["handler_function"],
        is_async=task_config["is_async"]
    ))




@app.post("/run_pipeline/")
async def run_pipeline(request: TaskRequest):
    task_name = request.task.lower()
    logging.info(request)
    try:
        task_config = task_loader.get_task(task_name)
        task_config.validate_inputs(request.model_dump())
        pipeline = TOOLS[task_config.pipeline_key]

        if task_config.is_async:
            output = await task_config.handler_function(request, pipeline)
        else:
            output = task_config.handler_function(request, pipeline)
        return output
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web_search/")
async def web_search(request: SearchRequest):
    task_name = request.task.lower()
    try:
        task_config = task_loader.get_task(task_name)
        task_config.validate_inputs(request.model_dump())
        requester = TOOLS[task_config.pipeline_key]

        if task_config.is_async:
            output = await task_config.handler_function(request, requester)
        else:
            output = task_config.handler_function(request, requester)
        return output
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def ping():
    return "Service available"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
