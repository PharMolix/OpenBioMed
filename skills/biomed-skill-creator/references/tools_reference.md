# OpenBioMed Tools Reference

This document provides a comprehensive reference for all available tools in OpenBioMed.

## Tool Registry Access

```python
from open_biomed.tools import TOOLS

# List all available tools
print(TOOLS.available_tools())

# Access a specific tool
tool = TOOLS["tool_name"]
result, message = tool.run(**kwargs)
```

## Molecular Tools

### Data Retrieval

#### molecule_name_request
Retrieves molecule from PubChem by common name.

```python
tool = TOOLS["molecule_name_request"]
result, message = tool.run(name="aspirin")
molecule = result["molecule"]
# Returns: Molecule object with SMILES, properties
```

#### pubchemid_search
Searches PubChem by CID.

```python
tool = TOOLS["pubchemid_search"]
result, message = tool.run(cid=2244)  # Aspirin CID
```

#### molecule_structure_request
Finds similar molecules in PubChem.

```python
tool = TOOLS["molecule_structure_request"]
result, message = tool.run(smiles="CC(=O)OC1=CC=CC=C1C(=O)O", similarity_threshold=0.9)
```

### Property Calculation

#### molecule_qed
Calculates QED (Quantitative Estimate of Drug-likeness).

```python
tool = TOOLS["molecule_qed"]
result, message = tool.run(molecule=mol)
qed_score = result.get("qed", result)
# Range: 0-1, higher is better
# > 0.7: Excellent, 0.5-0.7: Good, < 0.5: Poor
```

#### molecule_sa
Calculates Synthetic Accessibility score.

```python
tool = TOOLS["molecule_sa"]
result, message = tool.run(molecule=mol)
sa_score = result.get("sa", result)
# Range: 1-10, lower is easier
# 1-3: Easy, 3-6: Moderate, 6-10: Difficult
```

#### molecule_logp
Calculates lipophilicity (LogP).

```python
tool = TOOLS["molecule_logp"]
result, message = tool.run(molecule=mol)
logp = result.get("logp", result)
# Optimal range: -0.4 to 5.6 for oral drugs
```

#### molecule_lipinski
Counts Lipinski's Rule of Five violations.

```python
tool = TOOLS["molecule_lipinski"]
result, message = tool.run(molecule=mol)
violations = result.get("violations", result)
# 0 violations = ideal, 1 = acceptable, 2+ = concerning
```

#### molecule_similarity
Calculates Tanimoto similarity between two molecules.

```python
tool = TOOLS["molecule_similarity"]
result, message = tool.run(molecule_1=mol1, molecule_2=mol2)
similarity = result[0]  # Range: 0-1
```

### Property Prediction

#### molecule_property_prediction
Predicts ADMET properties using GraphMVP.

```python
tool = TOOLS["molecule_property_prediction"]

# Blood-brain barrier penetration
result, msg = tool.run(molecule=mol, dataset="bbbp", model="graphmvp")
# Returns: 0 (no penetration) or 1 (penetrates)

# Side effects (SIDER - 27 categories)
result, msg = tool.run(molecule=mol, dataset="sider", model="graphmvp")
# Returns: Dict of side effect categories and predictions
```

### Question Answering

#### molecule_question_answering
Answers questions about molecules using BioT5.

```python
tool = TOOLS["molecule_question_answering"]
result, message = tool.run(molecule=mol, text="What functional groups does this molecule have?")
answer = result["answer"]
```

### Editing

#### text_based_molecule_editing
Edits molecules based on text instructions.

```python
# Requires InferencePipeline
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="text_based_molecule_editing",
    model="biot5_plus",
    device="cuda:0"
)
result = pipeline.run(molecule=mol, text="reduce toxicity")
```

### Visualization

#### visualize_molecule
Creates 2D/3D visualization of molecules.

```python
tool = TOOLS["visualize_molecule"]
result, message = tool.run(
    molecule=mol,
    style="ball_stick",  # Options: ball_stick, stick, line, sphere
    show_hydrogen=False
)
```

### Export

#### export_molecule
Exports molecule to SDF format.

```python
tool = TOOLS["export_molecule"]
result, message = tool.run(molecule=mol, output_path="molecule.sdf")
```

## Protein Tools

### Data Retrieval

#### protein_uniprot_request
Retrieves protein sequence from UniProt.

```python
tool = TOOLS["protein_uniprot_request"]
result, message = tool.run(accession="P00533")  # EGFR
protein = result["protein"]
```

#### protein_pdb_request
Downloads PDB or AlphaFold structure.

```python
tool = TOOLS["protein_pdb_request"]
result, message = tool.run(accession="1ABC", mode="file_only")
pdb_path = result["pdb_file"]
```

### Structure Analysis

#### protein_folding
Predicts 3D structure using ESMFold.

```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="protein_folding",
    model="esmfold",
    device="cuda:0"
)
result = pipeline.run(protein=protein)
```

#### protein_binding_site_prediction
Predicts binding sites using P2Rank.

```python
tool = TOOLS["protein_binding_site_prediction"]
result, message = tool.run(protein=protein)
# Returns list of predicted binding sites with scores
```

### Mutation Analysis

#### mutation_explanation
Explains effects of single-site mutations using MutaPLM.

```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="mutation_explanation",
    model="mutaplm",
    device="cuda:0"
)
result = pipeline.run(protein=protein, mutation="A123V")
```

#### mutation_engineering
Generates mutations based on text description.

```python
pipeline = InferencePipeline(
    task="mutation_engineering",
    model="mutaplm",
    device="cuda:0"
)
result = pipeline.run(protein=protein, text="increase stability")
```

### Question Answering

#### protein_question_answering
Answers questions about proteins.

```python
tool = TOOLS["protein_question_answering"]
result, message = tool.run(protein=protein, text="What is the function of this protein?")
```

### Visualization

#### visualize_protein
Creates protein structure visualization.

```python
tool = TOOLS["visualize_protein"]
result, message = tool.run(
    protein=protein,
    style="cartoon"  # Options: cartoon, surface, all-atom
)
```

## Pocket Tools

### Pocket Creation

#### import_pocket
Creates pocket from protein subsequence.

```python
tool = TOOLS["import_pocket"]
result, message = tool.run(protein=protein, residue_indices=[100, 101, 102, ...])
pocket = result["pocket"]
```

### Visualization

#### visualize_protein_pocket
Visualizes protein binding pocket.

```python
tool = TOOLS["visualize_protein_pocket"]
result, message = tool.run(protein=protein, pocket=pocket)
```

## Drug Design Tools

### Structure-Based Design

#### structure_based_drug_design
Generates molecules for a binding pocket using MolCraft or PharmolixFM.

```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="structure_based_drug_design",
    model="molcraft",
    model_ckpt="./checkpoints/molcraft/last_updated.ckpt",
    device="cuda:0"
)
candidates = pipeline.run(pocket=pocket, num_samples=10, num_sample_steps=100)
```

### Docking

#### pocket_molecule_docking
Docks molecules into pockets using PharmolixFM.

```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="pocket_molecule_docking",
    model="pharmolixfm",
    device="cuda:0"
)
result = pipeline.run(pocket=pocket, molecule=mol)
```

#### protein_molecule_docking_score
Calculates docking score using AutoDock Vina.

```python
tool = TOOLS["protein_molecule_docking_score"]
result, message = tool.run(protein=protein, molecule=mol)
docking_score = result["score"]  # kcal/mol, more negative = better binding
```

### File Processing

#### extract_molecules_from_pdb_file
Extracts molecules, proteins, and ions from PDB files.

```python
tool = TOOLS["extract_molecules_from_pdb_file"]
result, message = tool.run(pdb_file="structure.pdb")
# Returns: List of (type, chain_id, entity) tuples
```

## Utility Tools

### Web Search

#### web_search
Searches web for biomedical information.

```python
tool = TOOLS["web_search"]
result, message = tool.run(query="aspirin mechanism of action")
```

### Summarization

#### summarize_content
Summarizes text content using LLM.

```python
tool = TOOLS["summarize_content"]
result, message = tool.run(text=long_text, max_length=200)
```

### Export

#### export_protein
Exports protein to PDB format.

```python
tool = TOOLS["export_protein"]
result, message = tool.run(protein=protein, output_path="protein.pdb")
```

## Inference Pipeline

For model-based tasks, use the InferencePipeline:

```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="task_name",           # Required
    model="model_name",         # Required
    model_ckpt="path/to/ckpt",  # Optional, uses default if not provided
    device="cuda:0"             # Optional, defaults to cuda:0
)

# Run inference
result = pipeline.run(**inputs)
```

### Available Tasks and Models

| Task | Models |
|------|--------|
| molecule_property_prediction | graphmvp, graphmvp_regression |
| molecule_question_answering | molt5, biot5, biot5_plus |
| protein_question_answering | molt5, biot5, biot5_plus |
| text_based_molecule_editing | molt5, biot5, biot5_plus, llm4molopt |
| structure_based_drug_design | pharmolixfm, molcraft |
| pocket_molecule_docking | pharmolixfm |
| mutation_explanation | mutaplm |
| mutation_engineering | mutaplm |
| protein_folding | esmfold |
| cell_annotation | langcell |

## Error Handling

```python
try:
    result, message = tool.run(**kwargs)
    if result is None:
        print(f"Tool returned no result: {message}")
except Exception as e:
    print(f"Error running tool: {e}")
```

Common errors:
- **Molecule not found**: Name not in PubChem database
- **Model not found**: Checkpoint file missing
- **CUDA out of memory**: Reduce batch size or use CPU
- **Network timeout**: PubChem/UniProt API slow or down
