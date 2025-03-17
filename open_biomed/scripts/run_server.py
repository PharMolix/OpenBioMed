from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn.functional as F
import uvicorn
import random
import copy
import asyncio
import subprocess
from typing import Optional, List, Dict, Callable, Any

# import function
from open_biomed.data import Molecule, Text, Protein, Pocket
from open_biomed.core.tool_misc import MutationToSequence
from open_biomed.core.oss_warpper import oss_warpper
from open_biomed.core.tool_registry import TOOLS


app = FastAPI()


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
    visualize_config: Optional[str] = None
    molecule: Optional[str] = None
    protein: Optional[str] = None
    pocket: Optional[str] = None
    text: Optional[str] = None
    dataset: Optional[str] = None
    query: Optional[str] = None
    mutation: Optional[str] = None
    indices: Optional[str] = None


class SearchRequest(BaseModel):
    task: str
    query: Optional[str] = None
    molecule: Optional[str] = None
    threshold: Optional[str] = None


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
                    "python3", "./open_biomed/core/visualize.py", 
                    "--task", "visualize_molecule",
                    "--molecule_config", request.visualize_config,
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
                    "python3", "./open_biomed/core/visualize.py", 
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
    outputs = await requester.run(request.query)
    smiles = outputs[0][0].smiles
    output = outputs[1][0]
    return {"task": request.task, "molecule": output, "molecule_preview": smiles}

def handle_web_search(request: SearchRequest, requester):
    outputs = requester.run(request.query)
    outputs = outputs[0][0]
    return {"task": request.task, "text": outputs}

async def handle_molecule_structure_request(request: SearchRequest, requester):
    molecule = IO_Reader.get_molecule(request.molecule)
    threshold = request.threshold
    outputs = await requester.run(molecule, threshold=float(threshold), max_records=1)
    # Pick a random molecule
    index = random.randint(0, len(outputs[1])-1)
    molecule = outputs[1][index]
    molecule_preview = outputs[0][index].smiles
    return {"task": request.task, "molecule": molecule, "molecule_preview": molecule_preview}


async def handle_protein_uniprot_request(request: SearchRequest, requester):
    outputs = await requester.run(request.query)
    outputs = outputs[1][0]
    return {"task": request.task, "text": outputs}


async def handle_protein_pdb_request(request: SearchRequest, requester):
    outputs = await requester.run(request.query)
    outputs = outputs[1][0]
    protein = IO_Reader.get_protein(outputs)
    protein_preview = str(protein)
    return {"task": request.task, "protein": outputs, "protein_preview": protein_preview}


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
                    "python3", "./open_biomed/core/visualize.py", 
                    "--task", "visualize_protein",
                    "--protein_config", request.config,
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
                    "python3", "./open_biomed/core/visualize.py", 
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
        "required_inputs": ["visualize_config", "molecule"],
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
        "required_inputs": ["protein"],
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
        "is_async": False
    },
    {
        "task_name": "molecule_structure_request",
        "required_inputs": ["molecule", "threshold"],
        "pipeline_key": "molecule_structure_request",
        "handler_function": handle_molecule_structure_request,
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

    try:
        task_config = task_loader.get_task(task_name)
        task_config.validate_inputs(request.model_dump())
        pipeline = TOOLS[task_config.pipeline_key]
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



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
