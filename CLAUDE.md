# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenBioMed is a Python deep learning toolkit for AI-empowered biomedicine. It provides flexible APIs for multi-modal biomedical data (molecules, proteins, pockets, cells, text) and includes 20+ tools for downstream applications including drug discovery, protein engineering, and multi-modal reasoning.

## Key Commands

### Environment Setup
```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
# Using shell script
./scripts/train.sh TASK MODEL DATASET GPU_ID

# Using Python directly
python open_biomed/scripts/train.py \
    --task TASK \
    --additional_config_file configs/model/MODEL.yaml \
    --dataset_name DATASET \
    --dataset_path ./datasets/TASK/DATASET
```

### Testing/Evaluation
```bash
./scripts/test.sh TASK MODEL DATASET GPU_ID
```

### Inference
```bash
python open_biomed/scripts/inference.py --task TASK_NAME
```

### Running Server
```bash
python -m uvicorn open_biomed.scripts.run_server:app --host 0.0.0.0 --port 8082
python -m uvicorn open_biomed.scripts.run_server_workflow:app --host 0.0.0.0 --port 8083
```

## Architecture

### Core Directories
- `open_biomed/data/`: Data structures for `Molecule`, `Protein`, `Pocket`, `Cell`, `Text` entities
- `open_biomed/models/`: Model implementations organized by type
  - `foundation_models/`: BioT5, MolT5, PharmolixFM, BioMedGPT, etc.
  - `task_models/`: Task-specific model wrappers
  - `protein/`, `molecule/`, `cell/`: Domain-specific models
- `open_biomed/tasks/`: Task definitions
  - `aidd_tasks/`: Drug discovery tasks (property prediction, docking, drug design)
  - `multi_modal_tasks/`: QA, captioning, translation tasks
- `open_biomed/tools/`: Tool implementations (visualization, web requests, property calculators)
- `open_biomed/core/`: Infrastructure
  - `pipeline.py`: `TrainValPipeline`, `InferencePipeline` for training/inference
  - `workflow.py`: DAG-based workflow execution
  - `agent.py`: LangGraph-based LLM agent system
- `configs/`: YAML configurations for models, datasets, workflows, visualization

### Key Registries
- `TASK_REGISTRY` (tasks/__init__.py): Maps task names to task classes
- `MODEL_REGISTRY` (models/__init__.py): Maps task names to available models
- `TOOLS` (tools/tool_registry.py): Lazy-loaded dictionary of available tools

### Data Entities
Create entities using factory methods:
```python
from open_biomed.data import Molecule, Protein, Pocket, Text

molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
molecule = Molecule.from_sdf_file("ligand.sdf")
protein = Protein.from_fasta("MKFLILLFNILCLFPVLAADNH...")
protein = Protein.from_pdb_file("protein.pdb")
pocket = Pocket.from_protein_ref_ligand(protein, ligand)
text = Text.from_str("Describe this molecule")
```

### Configuration System
- Uses YAML files with variable substitution (`!SUB ${var}`)
- Config files in `configs/model/`, `configs/dataset/`, `configs/workflow/`
- Merge configs using `merge_config()` for hierarchical configuration

### Workflow System
Workflows are defined in YAML with a DAG structure:
```yaml
tools:
  - name: molecule_name_request
    inputs:
      accession: aspirin
  - name: molecule_question_answering
    inputs:
      text: What is this molecule?
edges:
  - start: 0
    end: 1
```
Workflows are loaded from `memory/workflows/` and executed via `Workflow` class.

### Inference Pipeline
```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="molecule_property_prediction",
    model="graphmvp",
    model_ckpt="./checkpoints/model.ckpt",
    device="cuda:0"
)
outputs = pipeline.run(molecule=molecule)
```

## Available Tasks and Models

| Task | Available Models |
|------|------------------|
| molecule_property_prediction | graphmvp, graphmvp_regression |
| molecule_question_answering | molt5, biot5, biot5_plus |
| protein_question_answering | molt5, biot5, biot5_plus |
| text_based_molecule_editing | molt5, biot5, biot5_plus, llm4molopt |
| structure_based_drug_design | pharmolix_fm, molcraft |
| pocket_molecule_docking | pharmolix_fm |
| mutation_explanation | mutaplm |
| mutation_engineering | mutaplm |
| protein_folding | esmfold |
| cell_annotation | langcell |

## Model Checkpoints

Model checkpoints are stored in `./checkpoints/`. Pre-trained models are available:
- PharmolixFM: https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/
- BioMedGPT-R1: https://huggingface.co/PharMolix/BioMedGPT-R1
- Other models: See README.md for download links

## Coding Patterns

### Adding a New Task
1. Create task class in `open_biomed/tasks/` extending `BaseTask`
2. Register in `TASK_REGISTRY` in `tasks/__init__.py`
3. Create dataset class in `open_biomed/datasets/`
4. Add model support in `MODEL_REGISTRY` in `models/__init__.py`

### Adding a New Tool
1. Create tool class extending `Tool` in `open_biomed/tools/`
2. Register in `LazyDictForTool` in `tools/tool_registry.py`
3. Add to `available_tools()` list

### Adding a New Model
1. Create model class in appropriate subdirectory under `models/`
2. Add config file in `configs/model/`
3. Register in `MODEL_REGISTRY` mapping to relevant tasks
