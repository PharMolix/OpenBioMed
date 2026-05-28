# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenBioMed is a Python deep learning toolkit for AI-empowered biomedicine. It provides flexible APIs for multi-modal biomedical data (molecules, proteins, pockets, cells, text) and includes 20+ tools for downstream applications including drug discovery, protein engineering, and multi-modal reasoning. It also provides 45 skills (in `skills/`) for end-to-end biomedical research tasks powered by Claude Code.

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
# Using shell script (4 positional args)
./scripts/train.sh TASK MODEL DATASET GPU_ID

# Using Python directly
python open_biomed/scripts/train.py \
    --task TASK \
    --additional_config_file configs/model/MODEL.yaml \
    --dataset_name DATASET \
    --dataset_path ./datasets/TASK/DATASET
```

### Inference
```bash
python open_biomed/scripts/inference.py --task TASK_NAME
```

### Running Server
```bash
# Both servers together
sh ./scripts/run_server.sh

# Individually (ports configurable in run_server.sh)
python -m uvicorn open_biomed.scripts.run_server:app --host 0.0.0.0 --port 8095
python -m uvicorn open_biomed.scripts.run_server_workflow:app --host 0.0.0.0 --port 8094
```

### API Endpoints
- `/run_pipeline/` — Inference tasks (molecule editing, drug design, property prediction, etc.)
- `/web_search/` — Data retrieval (PubChem, UniProt, PDB, web search)

### Integration Testing
```bash
# Requires server running (default: http://127.0.0.1:8090)
python test/api_test.py --url http://127.0.0.1:8090
# Filter specific tests
python test/api_test.py --url http://127.0.0.1:8090 --test molecule
```

### Evaluation (using trained models)
```bash
./scripts/test.sh TASK MODEL DATASET GPU_ID
# Note: test.sh runs evaluation mode (test_only flag) with a hardcoded checkpoint path
```

## Architecture

### Core Directories
- `open_biomed/data/`: Data entity classes (`Molecule`, `Protein`, `Pocket`, `Cell`, `Text`, `KG`)
- `open_biomed/models/`: Model implementations
  - `foundation_models/`: BioT5, BioT5+, MolT5, PharmolixFM, BioMedGPT, etc.
  - `task_models/`: Task-specific wrappers (one per registered task)
  - `protein/`: MutaPLM, EsmFold, CodeFP
  - `molecule/`: GraphMVP, MolCRAFT
  - `cell/`: LangCell
  - `agentic_models/`: LLM-based molecule optimization
- `open_biomed/tasks/`: Task definitions (`base_task.py` has `BaseTask`, `ModelWrapper`, `DefaultDataModule`)
  - `aidd_tasks/`: Drug discovery tasks (property prediction, docking, drug design, DDI, PPI, protein design)
  - `multi_modal_tasks/`: QA, captioning, translation, mutation, protein generation tasks
- `open_biomed/datasets/`: Dataset implementations (`base_dataset.py`, 11 task-specific files)
- `open_biomed/tools/`: Tool implementations
  - `base_tool.py`: `Tool` ABC — `run()` returns `Tuple[List[Any], List[Any]]`; has `serial_exec` decorator for auto-batch iteration
  - `tool_registry.py`: `LazyDictForTool` / `TOOLS` — tools lazy-instantiate on first access
  - `tool_misc.py`: Inference pipeline-based tools and utility tools (property calculators, mutation-to-sequence, import/export)
  - `web_request_tools.py`: PubChem, UniProt, PDB, STRING, ChEMBL, web search requesters
  - `visualization_tools.py`: PyMol-based visualization wrappers
  - `third_party_tools.py`: Third-party tool integrations
- `open_biomed/core/`: Infrastructure
  - `pipeline.py`: `TrainValPipeline` (Lightning), `InferencePipeline` (dual-inherits `Pipeline` + `Tool` — usable as both), `EnsemblePipeline`
  - `workflow.py`: `Workflow` (DAG-based, topological sort), `WorkflowNode`, `WORKFLOWS` singleton (auto-loads from `memory/workflows/`)
  - `agent.py`: `PlannerExecutor` (LangGraph agent — generates/executes Python/bash code, supports Docker execution, checkpointing, plan tracking, report export, workflow export)
  - `llm_provider.py`: Multi-provider LLM dispatch (Claude, OpenAI, Gemini, DeepSeek, BioMedGPT custom)
  - `context_manager.py`: `ContextManager`, `ToolContextManager` — conversation history management for agents
- `open_biomed/utils/`: `config.py` (Config system with `!SUB ${var}` substitution), `featurizer.py`, `collator.py`, `misc.py` (`create_tool_input()` universal factory), `callbacks.py` (`RecoverCallback`, `GradientClip`)
- `configs/`: YAML configurations — `basic_config.yaml` (base), `model/`, `dataset/`, `workflow/`, `agent/`, `visualization/`
- `skills/`: 46 Claude Code skills organized by category (drug discovery, protein engineering, single-cell omics, data retrieval, utilities)

### Key Registries

All registries use nested dict structures `{task_name: {item_name: Class}}`:

- `TASK_REGISTRY` (`tasks/__init__.py`): Maps 16 task names to task classes
- `MODEL_REGISTRY` (`models/__init__.py`): Maps task names → model names → model classes
- `DATASET_REGISTRY` (`datasets/__init__.py`): Maps task names → dataset names → dataset classes (20+ datasets for molecule property prediction)
- `TOOLS` (`tools/tool_registry.py`): `LazyDictForTool` with 28 registered tool names; lazy-instantiates via `__missing__()` on first access

### Data Entities

Create entities using factory methods:
```python
from open_biomed.data import Molecule, Protein, Pocket, Cell, Text

molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
molecule = Molecule.from_sdf_file("ligand.sdf")
protein = Protein.from_fasta("MKFLILLFNILCLFPVLAADNH...")
protein = Protein.from_pdb_file("protein.pdb")
pocket = Pocket.from_protein_ref_ligand(protein, ligand)  # extracts residues near reference ligand
text = Text.from_str("Describe this molecule")
```

Universal input factory for agents/workflows:
```python
from open_biomed.utils.misc import create_tool_input
entity = create_tool_input("molecule", "CC(=O)OC1=CC=CC=C1C(=O)O")  # auto-detects format
```

### Configuration System
- Custom YAML extension: `!SUB ${var}` for variable substitution, parsed by `parse_config()` in `utils/config.py`
- Config merging: starts from `configs/basic_config.yaml`, then merges `--additional_config_file` args via `merge_config()`
- Hierarchical: model configs override dataset configs override basic config

### Workflow System

Workflows are DAGs of tool nodes with topological-sort execution:
```yaml
metadata:
  name: workflow_name
  inputs: [(tool_id, input_key, description)]
  outputs: [(tool_id, output_key, description)]
tools:
  - name: molecule_name_request
    inputs: {accession: aspirin}
    num_repeats: 1  # optional
  - name: code_execution  # arbitrary Python
    code: "result = step_inputs[0] + step_outputs[1]"
edges:
  - start: 0
    end: 1
    name_mapping: {output_key: input_key}  # optional key remapping
```
- `code_execution` nodes use `step_inputs[i]` and `step_outputs[i]` to access other nodes' data
- Workflows auto-loaded from `memory/workflows/` via `WORKFLOWS` singleton
- `parse_frontend()` converts LangFlow-style frontend JSON into workflow YAML

### Agent System

`PlannerExecutor` is the main LangGraph-based agent:
- Plan styles: `checklist` or `step-by-step`
- Generates `<execute>` blocks (Python/bash code) and `<report>` blocks (markdown)
- Supports Docker execution, checkpointing (SqliteSaver), persistent namespace (pickle)
- Monkey-patches matplotlib, Molecule/Protein save, and PyMol to capture output files for reports
- `export_as_workflow()` converts agent trajectories into reusable YAML workflows
- Config at `configs/agent/planner_executor.yaml` (defaults to `deepseek-reasoner`)

LLM provider dispatch (`core/llm_provider.py`):
- Priority: `API_KEY` + `API_URL` env vars (custom/self-hosted) > platform providers by model prefix
- Model prefix routing: `claude-*` → Anthropic, `openai-*` → OpenAI, `gemini-*` → OpenAI-compatible, `deepseek-*` → DeepSeek, `BioMedGPT*` → custom local model
- Default model: `claude-sonnet-4-20250514` (can override via `MODEL_NAME` env var)
- Agent config at `configs/agent/planner_executor.yaml` may specify different defaults for agent workflows

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
# InferencePipeline also inherits from Tool — usable in workflows
# Has auto-batch-size detection (halves on OOM), retry logic, saves to ./tmp/
```

## Available Tasks and Models

| Task | Available Models |
|------|------------------|
| molecule_property_prediction | graphmvp, graphmvp_regression |
| molecule_property_prediction_regression | graphmvp_regression |
| molecule_question_answering | molt5, biot5, biot5_plus |
| protein_question_answering | molt5, biot5, biot5_plus |
| text_based_molecule_editing | molt5, biot5, biot5_plus, llm4molopt |
| structure_text_based_molecule_optimization | llm4molopt |
| structure_based_drug_design | pharmolix_fm, molcraft |
| pocket_molecule_docking | pharmolix_fm |
| mutation_explanation | mutaplm |
| mutation_engineering | mutaplm |
| protein_folding | esmfold |
| cell_annotation | langcell |
| go_guided_protein_generation | codefp |

Note: Several task files exist in `aidd_tasks/` that are not yet in `TASK_REGISTRY` (drug_cell_response_prediction, drug_drug_interaction, protein_design, protein_protein_interaction, protein_property_prediction) — these are under development.

## Model Checkpoints

Model checkpoints stored in `./checkpoints/`. Download links:
- PharmolixFM: https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/
- BioMedGPT-R1: https://huggingface.co/PharMolix/BioMedGPT-R1
- Other models: See README.md for download links

## Coding Patterns

### Adding a New Task
1. Create task class in `open_biomed/tasks/` extending `BaseTask`
2. Register in `TASK_REGISTRY` in `tasks/__init__.py`
3. Create dataset class in `open_biomed/datasets/` and register in `DATASET_REGISTRY`
4. Create model wrapper in `open_biomed/models/task_models/`
5. Add model config in `configs/model/`
6. Register model in `MODEL_REGISTRY` mapping task → model → class

### Adding a New Tool
1. Create tool class extending `Tool` in `open_biomed/tools/`
2. Use `serial_exec` decorator if the tool should auto-iterate over list inputs
3. Register in `LazyDictForTool` in `tools/tool_registry.py` — add to both `__missing__()` factory and `available_tools()` list

### Adding a New Model
1. Create model class in appropriate subdirectory under `models/`
2. Add config file in `configs/model/`
3. Register in `MODEL_REGISTRY` mapping task → model → class
4. Create corresponding `ModelWrapper` in `task_models/`

### Package Note
`setup.py` registers package as `openbiomed` (no underscore) but source directory is `open_biomed` (with underscore). `find_packages` handles the mapping — imports use `open_biomed`.

## Skill Refactoring Guidelines

### Background
Recent commits have been refactoring skills to use the `run_pipeline` and `web_search` APIs. Key changes include:

**Refactored Skills (19 total)**:
- `target-based-lead-design`, `drug-candidate-discovery`, `pubchem-query`, `admet-prediction`
- `text-based-molecule-editing`, `disease-drug-intelligence`, `kegg-query`, `chembl-query`
- `molecule-biochemical-significance-query-biot5`, `biomedical-literature-search`
- `target-drug-report`, `iupac-name-identification-biot5`, `drug-lead-analysis`
- `uniprot-query`, `ppi-string-query`, `drug-drug-interaction-analysis`
- `retrosynthesis-planning`, `biomed-skill-creator`, `biomed-skill-router`

**New Tasks Added**: `literature_search`, `ddi_analysis`, `disease_drug_intel`

**Infrastructure Updates**:
- Async support added to `run_pipeline` endpoint
- Web search handlers converted to async
- Numpy type conversion fixes
- WebSearchRequester migrated to Alibaba Cloud IQS

### Refactoring Rules

1. **Use run_pipeline API**: Skills should call the `/run_pipeline/` endpoint to execute tasks. Use existing tasks and tools whenever possible.

2. **Reuse Before Adding**: Only create new tasks or tools when existing ones cannot meet the skill's requirements. Check `TASK_REGISTRY` and `TOOLS` before implementing new code.

3. **Testing Environment**: Services run in Docker containers. See `scripts/run_docker.sh` for container configuration.

4. **Unit Tests Required**: When adding new tasks, tools, or Python code, add comprehensive unit tests in `test/` directory.

5. **Validation**: After refactoring a skill, simulate user usage to test the skill. The refactored skill must produce results consistent with the original implementation before the refactoring is considered complete.