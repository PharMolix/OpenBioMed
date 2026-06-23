from fastapi import FastAPI, HTTPException, UploadFile, File, Header, BackgroundTasks
from pydantic import BaseModel
import torch.nn.functional as F
import uvicorn
import random
import copy
import asyncio
import logging
import subprocess
import os
import sys
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
# NOTE: uvicorn applies its own logging config when it loads the app, which
# reformats the OpenBioMed logger to "LEVEL:NAME:MESSAGE" (dropping the
# timestamp). To keep timestamps stable, we attach our own handler to the
# app logger and disable propagation so uvicorn never touches the format.
_log_formatter = logging.Formatter(
    fmt='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.basicConfig(level=logging.INFO, format=_log_formatter._fmt, datefmt=_log_formatter.datefmt)
logger = logging.getLogger('OpenBioMed')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(_log_formatter)
    logger.addHandler(_h)

# File upload configuration
# UPLOAD_API_KEY: middleware专属密钥，用于middleware→云服务之间的内部认证
# 用户认证由Open WebUI middleware层负责，用户不接触此key
UPLOAD_DIR = "./tmp/uploads"
MAX_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXT = {".pdb", ".sdf", ".mol", ".mol2", ".smi", ".pkl", ".csv", ".txt"}
UPLOAD_API_KEY = os.environ.get("UPLOAD_API_KEY", "")

def cleanup_old_uploads():
    """Remove uploaded files older than 24 hours."""
    now = time.time()
    max_age = 24 * 3600  # 24 hours
    if not os.path.exists(UPLOAD_DIR):
        return
    count = 0
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > max_age:
            os.remove(filepath)
            count += 1
    if count > 0:
        logger.info(f"Upload cleanup: removed {count} files older than 24h")


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
    color: Optional[Literal["grey", "spectrum"]] = None
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
    # Antibody structure prediction fields
    heavy_chain: Optional[str] = None  # Heavy chain FASTA sequence
    light_chain: Optional[str] = None  # Light chain FASTA sequence
    antigen: Optional[str] = None  # Antigen FASTA sequence (for complex mode)
    # Antibody design fields
    fasta: Optional[str] = None  # FASTA file path with design requirement
    antigen_pdb: Optional[str] = None  # Antigen PDB file path for antibody design
    epitope: Optional[str] = None  # Epitope residue numbers (space-separated)
    fasta_origin: Optional[str] = None  # Original antibody FASTA for affinity maturation
    num_samples: Optional[int] = None  # Number of samples per residue
    # Similar protein search fields
    search_type: Optional[str] = None  # "msa" or "foldseek"
    protein: Optional[str] = None  # FASTA sequence or PDB file path
    database: Optional[List[str]] = None  # FoldSeek databases
    # Mutation design AAV fields
    num_rounds: Optional[int] = None  # Number of optimization rounds
    population_size: Optional[int] = None  # Number of mutants per round
    max_mutations: Optional[int] = None  # Max point mutations per sequence
    diversity_weight: Optional[float] = None  # Weight for diversity in selection
    # tFold antibody structure prediction fields
    prediction_type: Optional[str] = None  # "antibody", "nanobody", "complex", or "epitope"
    output_name: Optional[str] = None  # Output file name
    msa_content: Optional[str] = None  # MSA content in a3m format (for complex)
    antigen_id: Optional[str] = None  # Chain ID for antigen (default: "A")
    pdb_file: Optional[str] = None  # PDB file path (for epitope)
    # IgGM antibody design fields
    design_type: Optional[str] = None  # "nanobody" or "heavy_light"
    heavy_chain_mask: Optional[str] = None  # Heavy chain with X for design regions
    light_chain_mask: Optional[str] = None  # Light chain with X (for heavy_light)
    steps: Optional[int] = None  # Sampling steps for IgGM design
    # Boltz2 structure prediction fields
    task_id: Optional[str] = None  # Project/batch ID for Boltz2
    task_name: Optional[str] = None  # Task name for Boltz2
    sequence: Optional[str] = None  # Protein sequence (for Boltz2 affinity)
    smiles: Optional[str] = None  # Ligand SMILES (for Boltz2 affinity)
    sequence_1: Optional[str] = None  # First protein sequence (for Boltz2 prot_complex)
    sequence_2: Optional[str] = None  # Second protein sequence (for Boltz2 prot_complex)


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
    if outputs[0][0] is None:
        raise HTTPException(status_code=500, detail="Model failed to generate a valid molecule. This may be due to missing dependencies (e.g., OpenBabel) or invalid pocket input.")
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


async def handle_protein_pdb_request(request: SearchRequest, requester):
    outputs = await requester.run_async(request.query, mode="file_only")
    pdb_path = outputs[0][0]
    protein = IO_Reader.get_protein(pdb_path)
    protein_file = protein.save_binary()
    protein_preview = str(protein)
    return {"task": request.task, "protein": protein_file, "protein_preview": protein_preview}


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
    if isinstance(output, tuple):
        score = output[0]
    else:
        score = output
    return {"task": request.task, "model":request.model, "score": str(score)}


def handle_visualize_protein(request: TaskRequest, pipeline):
    required_inputs = ["protein"]
    protein = IO_Reader.get_protein(request.protein)
    vis_process = [
                    "python3", "./open_biomed/tools/visualization_tools.py",
                    "--task", "visualize_protein",
                    "--protein_config", request.visualize,
                    "--save_output_filename", "./tmp/protein_visualization_file.txt",
                    "--protein", request.protein]
    if request.color:
        vis_process.extend(["--color", request.color])
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


async def handle_similar_protein_search(request: TaskRequest, pipeline):
    """
    Search for similar proteins using FoldSeek (structure similarity).
    MSA (sequence similarity) search is handled via direct API calls documented in skills/similar-protein-retrieval/SKILL.md.

    Inputs:
        - protein: PDB file path, PKL file path, or FASTA sequence (must exist on server for file paths)
        - database: (optional) List of FoldSeek databases

    Outputs:
        - result_path: Path to results file (.m8 for FoldSeek)
        - description: Description message
    """
    protein_input = request.protein
    database = request.database

    if not protein_input:
        raise ValueError("protein input is required")

    # Convert PKL file to PDB if needed (FoldSeek requires PDB file)
    import os
    if protein_input.endswith(".pkl") and os.path.exists(protein_input):
        protein_obj = Protein.from_binary_file(protein_input)
        protein_path = protein_obj.save_pdb()
        logger.info(f"Converted PKL to PDB: {protein_input} -> {protein_path}")
    else:
        protein_path = protein_input

    outputs, messages = await pipeline.run_async(
        protein=protein_path,
        database=database
    )

    return {
        "task": request.task,
        "result_path": outputs[0] if outputs else "",
        "description": messages[0]
    }


def handle_mutation_design_aav(request: TaskRequest, pipeline):
    """
    Design high-fitness AAV VP1 capsid protein mutants through multi-round optimization.

    Inputs:
        - num_rounds: Number of optimization rounds (default: 10)
        - population_size: Number of mutants per round (default: 96)
        - max_mutations: Max point mutations per sequence (default: 4)
        - diversity_weight: Weight for diversity in selection (default: 0.1)

    Outputs:
        - csv_file: Path to results CSV with top 96 mutants
        - description: Summary of optimization results
    """
    outputs, messages = pipeline.run(
        num_rounds=request.num_rounds or 10,
        population_size=request.population_size or 96,
        max_mutations=request.max_mutations or 4,
        diversity_weight=request.diversity_weight or 0.1
    )

    return {
        "task": request.task,
        "csv_file": outputs[0] if outputs else "",
        "description": messages[0]
    }


def handle_read_molecule_file(request: TaskRequest, pipeline):
    """
    Read molecule content from a file path.

    Inputs:
        - molecule_file: Path to molecule file (.pkl or .sdf)
        - include_sdf: (optional) Whether to include SDF content, default True

    Outputs:
        - smiles: SMILES string of the molecule
        - sdf_content: SDF file content (if include_sdf=True)
        - description: Description message
    """
    molecule_file = request.molecule
    include_sdf = request.value if request.value is not None else "true"
    include_sdf = include_sdf.lower() == "true"

    if not molecule_file:
        raise ValueError("molecule_file (molecule parameter) is required")

    outputs, messages = pipeline.run(
        molecule_file=molecule_file,
        include_sdf=include_sdf
    )

    return {
        "task": request.task,
        "smiles": outputs["smiles"],
        "name": outputs["name"],
        "sdf_content": outputs.get("sdf_content", ""),
        "description": messages
    }


def handle_read_protein_file(request: TaskRequest, pipeline):
    """
    Read protein content from a file path.

    Inputs:
        - protein_file: Path to protein file (.pkl or .pdb)
        - include_pdb: (optional) Whether to include PDB content, default True

    Outputs:
        - sequence: FASTA sequence of the protein
        - pdb_content: PDB file content (if include_pdb=True)
        - description: Description message
    """
    protein_file = request.protein
    include_pdb = request.value if request.value is not None else "true"
    include_pdb = include_pdb.lower() == "true"

    if not protein_file:
        raise ValueError("protein_file (protein parameter) is required")

    outputs, messages = pipeline.run(
        protein_file=protein_file,
        include_pdb=include_pdb
    )

    return {
        "task": request.task,
        "sequence": outputs["sequence"],
        "name": outputs["name"],
        "pdb_content": outputs.get("pdb_content", ""),
        "description": messages
    }


def handle_tfold_antibody_structure(request: TaskRequest, pipeline):
    """
    tFold antibody structure prediction.

    Supports:
    - antibody: Heavy + light chain structure prediction
    - nanobody: Single chain structure prediction
    - complex: Antigen-antibody complex prediction
    - epitope: Epitope residue determination from complex PDB

    Inputs:
        - prediction_type: "antibody", "nanobody", "complex", or "epitope"
        - heavy_chain: Heavy chain FASTA sequence
        - light_chain: Light chain FASTA sequence (for antibody/complex)
        - antigen: Antigen FASTA sequence (for complex)
        - antigen_id: Chain ID for antigen (default: "A")
        - msa_content: MSA content in a3m format (optional, for complex)
        - pdb_file: PDB file path (for epitope)
        - distance_cutoff: Distance threshold for epitope (default: 5.0)
        - output_name: Output file name (optional)

    Outputs:
        - pdb_file: Path to predicted PDB file (for structure tasks)
        - result_file: Path to JSON result file (for epitope)
        - description: Summary with confidence scores
    """
    outputs, messages = pipeline.run(
        prediction_type=request.prediction_type or "antibody",
        heavy_chain=request.heavy_chain,
        light_chain=request.light_chain,
        antigen=request.antigen,
        antigen_id=request.antigen_id or "A",
        msa_content=request.msa_content,
        pdb_file=request.pdb_file,
        distance_cutoff=request.distance_cutoff or 5.0,
        output_name=request.output_name
    )

    return {
        "task": request.task,
        "prediction_type": request.prediction_type,
        "result_path": outputs[0] if outputs else "",
        "description": messages[0]
    }


def handle_iggm_antibody_design(request: TaskRequest, pipeline):
    """
    IgGM antibody de novo design.

    Supports:
    - nanobody: Single chain antibody design
    - heavy_light: Full antibody design (two chains)

    Inputs:
        - design_type: "nanobody" or "heavy_light"
        - antigen_pdb: Antigen PDB file path
        - heavy_chain_mask: Heavy chain sequence with X for design regions
        - light_chain_mask: Light chain sequence with X (for heavy_light)
        - epitope: Epitope residue numbers (JSON list or comma-separated)
        - num_samples: Number of design samples (default 1)
        - steps: Sampling steps (default 10)
        - antigen_chain_id: Antigen chain ID (default "A")
        - output_name: Output file name prefix (optional)

    Outputs:
        - PDB files: Designed antibody structures
        - FASTA files: Designed sequences
        - JSON file: Complete result metadata
    """
    outputs, messages = pipeline.run(
        design_type=request.design_type or "nanobody",
        antigen_pdb=request.antigen_pdb,
        heavy_chain_mask=request.heavy_chain_mask,
        light_chain_mask=request.light_chain_mask,
        epitope=request.epitope,
        num_samples=request.num_samples or 1,
        steps=request.steps or 10,
        antigen_chain_id=request.antigen_id or "A",
        output_name=request.output_name
    )

    return {
        "task": request.task,
        "design_type": request.design_type,
        "output_files": outputs,
        "description": messages[0]
    }


def handle_boltz2_structure_prediction(request: TaskRequest, pipeline):
    """
    Boltz-2 structure and affinity prediction.

    Supports:
    - affinity: Protein-ligand affinity prediction (with structure)
    - prot_complex: Protein complex structure prediction

    Inputs:
        - prediction_type: "affinity" or "prot_complex"
        - task_id: Project/batch ID for directory organization
        - task_name: Task name for directory organization
        - sequence: Protein sequence (for affinity)
        - smiles: Ligand SMILES (for affinity)
        - sequence_1: First protein sequence (for prot_complex)
        - sequence_2: Second protein sequence (for prot_complex)
        - output_name: Output file name prefix (optional)

    Outputs:
        - PDB files: Predicted structures
        - JSON files: Affinity scores (for affinity mode)
        - description: Summary with prediction details
    """
    outputs, messages = pipeline.run(
        prediction_type=request.prediction_type or "affinity",
        task_id=request.task_id,
        task_name=request.task_name,
        sequence=request.sequence,
        smiles=request.smiles,
        sequence_1=request.sequence_1,
        sequence_2=request.sequence_2,
        output_name=request.output_name
    )

    return {
        "task": request.task,
        "prediction_type": request.prediction_type,
        "output_files": outputs,
        "description": messages[0]
    }


# 25
def handle_molecule_similarity(request: TaskRequest, pipeline):
    required_inputs = ["molecule_1", "molecule_2"]
    molecule_1 = IO_Reader.get_molecule(request.molecule_1)
    molecule_2 = IO_Reader.get_molecule(request.molecule_2)
    outputs = pipeline.run(molecule_1=molecule_1, molecule_2=molecule_2)
    similarity = outputs[0][0]
    return {"task": request.task, "model":request.model, "similarity": similarity}

# 26
def handle_molecule_property_calculation(request: TaskRequest, pipeline):
    required_inputs = ["molecule", "property"]
    molecule = IO_Reader.get_molecule(request.molecule)
    property = request.property
    outputs = pipeline.run(molecule=molecule, property=property)
    if isinstance(outputs, (int, float)):
        score = outputs
    elif isinstance(outputs, tuple):
        score = outputs[0][0]
    else:
        score = outputs
    return {"task": request.task, "model":request.model, "score": round(score, 5)}

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
    },
    {
        "task_name": "similar_protein_search",
        "required_inputs": ["protein"],
        "pipeline_key": "similar_protein_search",
        "handler_function": handle_similar_protein_search,
        "is_async": True
    },
    {
        "task_name": "mutation_design_aav",
        "required_inputs": [],
        "pipeline_key": "mutation_design_aav",
        "handler_function": handle_mutation_design_aav,
        "is_async": False
    },
    {
        "task_name": "read_molecule_file",
        "required_inputs": ["molecule"],
        "pipeline_key": "read_molecule_file",
        "handler_function": handle_read_molecule_file,
        "is_async": False
    },
    {
        "task_name": "read_protein_file",
        "required_inputs": ["protein"],
        "pipeline_key": "read_protein_file",
        "handler_function": handle_read_protein_file,
        "is_async": False
    },
    {
        "task_name": "tfold_antibody_structure",
        "required_inputs": [],
        "pipeline_key": "tfold_antibody_structure",
        "handler_function": handle_tfold_antibody_structure,
        "is_async": False
    },
    {
        "task_name": "iggm_antibody_design",
        "required_inputs": ["antigen_pdb", "heavy_chain_mask", "epitope"],
        "pipeline_key": "iggm_antibody_design",
        "handler_function": handle_iggm_antibody_design,
        "is_async": False
    },
    {
        "task_name": "boltz2_structure_prediction",
        "required_inputs": [],
        "pipeline_key": "boltz2_structure_prediction",
        "handler_function": handle_boltz2_structure_prediction,
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




@app.post("/run_pipeline/")
async def run_pipeline(request: TaskRequest):
    task_name = request.task.lower()
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    inputs_dump = request.model_dump(exclude={'task', 'model'}, exclude_none=True)
    logger.info("=" * 80)
    logger.info(f"[TASK] {task_name} | RequestID: {request_id} | model={request.model}")
    logger.info(f"[INPUT] {inputs_dump}")
    try:
        task_config = task_loader.get_task(task_name)
        task_config.validate_inputs(request.model_dump())
        pipeline = TOOLS[task_config.pipeline_key]

        logger.info(f"[START] Task: {task_name}, Model: {request.model}")
        logger.info("[EXEC] Running pipeline...")
        if task_config.is_async:
            output = await task_config.handler_function(request, pipeline)
        else:
            output = task_config.handler_function(request, pipeline)
        elapsed = time.time() - start_time
        logger.info(f"[DONE] Task: {task_name} | RequestID: {request_id} | completed in {elapsed:.2f}s")
        logger.info(f"[OUTPUT] {output}")
        logger.info("=" * 80)
        return output
    except HTTPException:
        elapsed = time.time() - start_time
        logger.error(f"[FAILED] Task: {task_name} | RequestID: {request_id} | in {elapsed:.2f}s (HTTPException)")
        logger.info("=" * 80)
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"[FAILED] Task: {task_name} | RequestID: {request_id} | in {elapsed:.2f}s")
        logger.error(f"[ERROR] Exception: {e}")
        logger.error(f"[TRACEBACK]\n{error_traceback}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web_search/")
async def web_search(request: SearchRequest):
    task_name = request.task.lower()
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    inputs_dump = request.model_dump(exclude={'task'}, exclude_none=True)
    logger.info("=" * 80)
    logger.info(f"[WEB_SEARCH] {task_name} | RequestID: {request_id}")
    logger.info(f"[INPUT] {inputs_dump}")
    try:
        task_config = task_loader.get_task(task_name)
        task_config.validate_inputs(request.model_dump())
        requester = TOOLS[task_config.pipeline_key]

        # Check NCBI rate limiter status for NCBI-related tasks
        from open_biomed.tools.web_request_tools import _ncbi_limiter
        ncbi_tasks = _ncbi_limiter.ncbi_tasks
        if task_name in ncbi_tasks:
            if _ncbi_limiter.is_in_cooldown:
                blocks = _ncbi_limiter.consecutive_blocks
                elapsed = time.time() - start_time
                logger.warning(f"[COOLDOWN] {task_name} | RequestID: {request_id} | NCBI blocked {blocks}x after {elapsed:.2f}s")
                logger.info("=" * 80)
                raise HTTPException(
                    status_code=503,
                    detail=f"NCBI API is temporarily blocked due to rate limiting. "
                           f"Please retry in a few minutes. (consecutive blocks: {blocks})"
                )

        logger.info(f"[START] Task: {task_name}")
        logger.info("[EXEC] Running requester...")
        if task_config.is_async:
            output = await task_config.handler_function(request, requester)
        else:
            output = task_config.handler_function(request, requester)
        elapsed = time.time() - start_time
        logger.info(f"[DONE] Task: {task_name} | RequestID: {request_id} | completed in {elapsed:.2f}s")
        logger.info(f"[OUTPUT] {output}")
        logger.info("=" * 80)
        return output
    except HTTPException:
        elapsed = time.time() - start_time
        logger.error(f"[FAILED] Task: {task_name} | RequestID: {request_id} | in {elapsed:.2f}s (HTTPException)")
        logger.info("=" * 80)
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        error_traceback = traceback.format_exc()
        error_str = str(e)
        logger.error(f"[FAILED] Task: {task_name} | RequestID: {request_id} | in {elapsed:.2f}s")
        logger.error(f"[ERROR] Exception: {error_str}")
        logger.error(f"[TRACEBACK]\n{error_traceback}")
        logger.info("=" * 80)
        # Check if this is an NCBI block error — return 503 instead of 500
        if "Blocked" in error_str or "rate-limited" in error_str or "HTML error page" in error_str:
            raise HTTPException(
                status_code=503,
                detail=f"NCBI API is temporarily unavailable: {error_str}. Please retry in a few minutes."
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def ping():
    return "Service available"

@app.on_event("startup")
async def startup_create_upload_dir():
    """Create upload directory on startup and run initial cleanup."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    cleanup_old_uploads()
    logger.info(f"Upload directory ready: {UPLOAD_DIR}")

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    """Upload a file to the server for Skill consumption.

    Requires X-API-Key header for authentication.
    Returns the absolute path where the file is stored.
    """
    # Auth check
    if not UPLOAD_API_KEY:
        raise HTTPException(status_code=403, detail="Upload API key not configured on server")
    if x_api_key != UPLOAD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # File type check
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Allowed types: {', '.join(sorted(ALLOWED_EXT))}"
        )

    # Size check — read content once, check size, then write
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) // (1024*1024)}MB, max {MAX_SIZE // (1024*1024)}MB)"
        )

    # UUID rename and save
    new_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(UPLOAD_DIR, new_name)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)

    # Schedule cleanup in background (runs after response is sent)
    background_tasks.add_task(cleanup_old_uploads)

    logger.info(f"Upload saved: {save_path} ({len(content)} bytes, original={file.filename})")
    return {"path": save_path, "filename": new_name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8095)
