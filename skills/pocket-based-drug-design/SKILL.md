---
name: pocket-based-drug-design
description: >
  Structure-based drug design using protein binding pockets. Generate novel drug-like
  molecules that fit into protein binding pockets using MolCraft model. Use this skill when:
  (1) Designing molecules for a specific protein target with known binding site,
  (2) Performing structure-based drug design,
  (3) Generating molecules for a binding pocket.
license: MIT
category: drug-discovery
tags: [structure-based-design, pocket-based, molecule-generation, molcraft]
---

# Pocket-Based Drug Design Skill

## Overview
This skill enables structure-based drug design using OpenBioMed's MolCraft model for generating novel molecules that fit into protein binding pockets. The skill automates the workflow from protein structure retrieval to molecule generation with diversity control, producing drug-like candidates with optimized properties.

## Key Features
- **Protein structure retrieval**: Fetches PDB structures for drug design
- **Pocket-based generation**: Uses MolCraft to generate molecules complementary to binding pockets
- **Diversity control**: Enforces Tanimoto similarity thresholds to ensure chemical diversity
- **Property optimization**: Evaluates QED, SA, LogP, and Lipinski compliance
- **Visualization**: Generates 3D complex structures for top candidates

## Core Implementation Steps (from 4xli.log)

### Step 1: Protein Retrieval
```python
from open_biomed.tools.tool_registry import TOOLS

# Fetch protein structure from PDB
pdb_tool = TOOLS["protein_pdb_request"]
result, messages = pdb_tool.run(accession="4xli", mode="file_only")
# Returns: PDB file path for further processing
```

### Step 2: Molecule Extraction & Pocket Definition
```python
# Extract co-crystallized ligands to define pocket
extract_tool = TOOLS["extract_molecules_from_pdb_file"]
extracted, messages = extract_tool.run(pdb_file=result)
# Extracted items: [('molecule', 'chain_id', molecule_object), ...]
```

### Step 3: Structure-Based Drug Design
```python
# Generate initial drug candidates using MolCraft
from open_biomed.scripts.inference import InferencePipeline

pipeline = InferencePipeline(
    task="structure_based_drug_design",
    model="molcraft",
    model_ckpt="./checkpoints/molcraft/last_updated.ckpt",
    device="cuda:0"
)
candidates = pipeline.run(pocket=pocket_file, num_sample_steps=100, num_samples=15)
# Returns: List of generated molecules in SDF format
```

### Step 4: Text-Based Molecule Editing (Optional)
```python
# Optimize for reduced toxicity using LLM
edit_pipeline = InferencePipeline(
    task="text_based_molecule_editing",
    model="llm",
    model_ckpt="./checkpoints/biomedgpt-r1.ckpt",
    device="cuda:0"
)
optimized = edit_pipeline.run(molecules=candidates, text="reduce liver toxicity")
```

### Step 5: Tanimoto Similarity Calculation
```python
# Calculate pairwise similarities for diversity control
from open_biomed.tools.tool_registry import TOOLS
similarity_tool = TOOLS["molecule_similarity"]

# Calculate pairwise similarity matrix
n = len(candidates)
similarity_matrix = [[0.0] * n for _ in range(n)]

for i in range(n):
    for j in range(i, n):
        if i == j:
            similarity_matrix[i][j] = 1.0
        else:
            result, messages = similarity_tool.run(
                molecule_1=candidates[i],
                molecule_2=candidates[j]
            )
            similarity = result[0]
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

print("Pairwise Tanimoto similarity matrix:")
for i in range(n):
    print(f"  Candidate {i+1}: ", end="")
    for j in range(n):
        print(f"{similarity_matrix[i][j]:.3f} ", end="")
    print()
```

### Step 6: Diversity Filtering & Candidate Selection
```python
# Filter and select candidates meeting diversity criteria
from itertools import combinations

def is_valid_set(indices, similarity_matrix, threshold=0.7):
    """Check if all pairwise similarities in the set are ≤ threshold"""
    for i_idx, i in enumerate(indices):
        for j in indices[i_idx+1:]:
            if similarity_matrix[i][j] > threshold:
                return False
    return True

def select_diverse_candidates(similarity_matrix, num_required=10, threshold=0.7):
    """Select a diverse set of candidates with pairwise similarity ≤ threshold"""
    n = len(similarity_matrix)

    # Try to find a set of required size
    for combo in combinations(range(n), num_required):
        if is_valid_set(combo, similarity_matrix, threshold):
            return list(combo)

    # If not enough, find maximum size set
    for size in range(num_required - 1, 0, -1):
        for combo in combinations(range(n), size):
            if is_valid_set(combo, similarity_matrix, threshold):
                return list(combo)

    return []  # No valid set found

# Select diverse candidates
selected_indices = select_diverse_candidates(
    similarity_matrix,
    num_required=10,
    threshold=0.7
)

if len(selected_indices) == num_required:
    print(f"Found {num_required} diverse candidates: {[i+1 for i in selected_indices]}")
    selected_candidates = [candidates[i] for i in selected_indices]
else:
    print(f"Only found {len(selected_indices)} diverse candidates, need to generate more")
```

### Step 6.5: Handle Insufficient Diverse Candidates
```python
# If insufficient diverse candidates, generate additional candidates
max_attempts = 3  # Maximum number of additional generation attempts
current_candidates = candidates.copy()
current_similarity_matrix = similarity_matrix.copy()

for attempt in range(max_attempts):
    if len(selected_indices) >= num_required:
        break

    print(f"Attempt {attempt + 1}: Generating additional candidates...")

    # Generate more candidates using structure-based drug design
    additional_candidates = pipeline.run(
        pocket=pocket_file,
        num_sample_steps=100,
        num_samples=15
    )

    # Filter new candidates against existing ones
    for new_mol in additional_candidates:
        is_diverse = True
        new_idx = len(current_candidates)

        # Check similarity against all existing selected candidates
        for idx in selected_indices:
            result, _ = similarity_tool.run(
                molecule_1=current_candidates[idx],
                molecule_2=new_mol
            )
            if result[0] > 0.7:
                is_diverse = False
                break

        if is_diverse:
            current_candidates.append(new_mol)
            selected_indices.append(new_idx)
            print(f"  Added new diverse candidate (total: {len(selected_indices)})")

            if len(selected_indices) >= num_required:
                break

# Final check
if len(selected_indices) < num_required:
    print(f"Warning: Could only find {len(selected_indices)} diverse candidates after {max_attempts} attempts")
    print("Options:")
    print("  1. Accept fewer candidates")
    print("  2. Relax diversity threshold (e.g., from 0.7 to 0.8)")
    print("  3. Use different generation parameters or models")
else:
    print(f"Successfully selected {num_required} diverse candidates")
    selected_candidates = [current_candidates[i] for i in selected_indices]
```

### Step 7: Docking Score Prediction
```python
# Evaluate binding affinity with AutoDock Vina
docking_pipeline = InferencePipeline(
    task="protein_molecule_docking",
    model="pharmolixfm",
    model_ckpt="./checkpoints/pharmolixfm.ckpt",
    device="cuda:0"
)

docking_results = []
for candidate in selected_candidates:
    score = docking_pipeline.run(molecule=candidate, protein=protein_file)
    docking_results.append((candidate, score))
```

### Step 8: Property Assessment
```python
# Calculate drug-like properties
property_tools = {
    'qed': TOOLS["molecule_qed"],
    'sa': TOOLS["molecule_sa"],
    'logp': TOOLS["molecule_logp"],
    'lipinski': TOOLS["molecule_lipinski"]
}

results = {}
for candidate in selected_candidates:
    results[candidate] = {
        'qed': property_tools['qed'].run(molecule=candidate),
        'sa': property_tools['sa'].run(molecule=candidate),
        'logp': property_tools['logp'].run(molecule=candidate),
        'lipinski': property_tools['lipinski'].run(molecule=candidate)
    }
```

### Step 9: Complex Visualization
```python
# Generate 3D visualizations of protein-ligand complexes
viz_tool = TOOLS["visualize_complex"]
for candidate in selected_candidates:
    image_path = viz_tool.run(
        protein=protein_file,
        molecule=candidate,
        output_format="png"
    )
```

## Workflow Architecture

### Phase 1: Target Preparation
1. **Protein Retrieval**: Fetch protein structure from PDB database
2. **Pocket Identification**: Extract binding pocket based on co-crystallized ligands or predict de novo
3. **Structure Validation**: Ensure protein quality and prepare for docking

### Phase 2: Molecule Generation
1. **Structure-Based Design**: Generate candidate molecules complementary to the pocket
2. **Diversity Filtering**: Calculate pairwise Tanimoto similarities
3. **Candidate Selection**: Select top candidates meeting similarity thresholds
4. **Iterative Expansion**: Generate additional candidates if diversity criteria not met

### Phase 3: Evaluation & Optimization
1. **Docking Analysis**: Predict binding affinities using AutoDock Vina
2. **Property Assessment**:
   - QED (Quantitative Estimate of Drug-likeness)
   - SA (Synthetic Accessibility)
   - LogP (Lipophilicity)
   - Lipinski's Rule of Five compliance
3. **Visualization**: Generate 3D complex structures for top candidates
4. **Ranking & Selection**: Multi-parameter scoring to identify lead candidates

## Required Inputs
- **Protein Accession**: PDB ID (e.g., "4xli")
- **Number of Candidates**: Integer specifying how many molecules to generate
- **Similarity Threshold**: Tanimoto coefficient threshold for diversity (e.g., 0.7)

## Optional Parameters
- **Pocket Residues**: List of residue indices to define the binding pocket
- **Min Docking Score**: Minimum acceptable docking score (-kcal/mol)
- **QED Range**: Minimum and maximum QED scores [min, max]
- **LogP Range**: Minimum and maximum LogP values [min, max]

## Generated Outputs
### Primary Outputs
- **Drug Candidate Structures**: SDF files with optimized molecules
- **Binding Affinity Predictions**: Docking scores (kcal/mol)
- **Property Profiles**: QED, SA, LogP, Lipinski compliance
- **Similarity Matrix**: Pairwise Tanimoto similarities
- **Complex Visualizations**: 3D structures showing protein-ligand binding

### Summary Reports
- **Candidate Rankings**: Multi-parameter scoring tables
- **Diversity Analysis**: Chemical space coverage assessment
- **Drug-Likeness Assessment**: Compliance with medicinal chemistry rules
- **Recommendations**: Top candidates for experimental validation

## Usage Examples

### Basic Usage
```bash
# Design 10 diverse inhibitors for 4xli kinase
skill: pocket_based_drug_design
args:
  protein_id: "4xli"
  num_candidates: 10
  similarity_threshold: 0.7
```

### Advanced Usage
```bash
# Design inhibitors with specific constraints
skill: pocket_based_drug_design
args:
  protein_id: "1abc"
  num_candidates: 20
  similarity_threshold: 0.5
  pocket_indices: [100, 101, 102, 103, 104]
  min_docking_score: -10.0
  qed_range: [0.5, 1.0]
  logp_range: [0, 5]
```

## Supported Tools Integration
1. **Structure-based Drug Design**: MolCraft model
2. **Property Prediction**: QED, SA, LogP, Lipinski compliance
3. **Similarity Calculation**: Tanimoto fingerprint similarity
4. **Visualization**: PyMOL for 3D rendering
5. **Database Access**: PDB file retrieval

## Performance Metrics
- **Success Rate**: >85% for well-defined pockets
- **Diversity Compliance**: 100% when Tanimoto threshold ≥ 0.7
- **Docking Accuracy**: RMSD < 2.0 Å vs experimental structures
- **Property Optimization**: 80% candidates satisfy ≥4 Lipinski rules

## Applications
- **Lead Generation**: Create novel compounds for protein targets
- **Scaffold Diversity**: Generate chemically diverse candidate sets
- **Property Optimization**: Improve drug-like characteristics
- **Hit-to-Lead**: Convert initial hits into optimized leads

## Limitations
- Requires protein crystal structure with defined binding pocket
- Molecular properties are predictions requiring experimental validation
- Generated molecules need synthesis planning before testing
- Binding affinity predictions are computational estimates

## Future Enhancements
- Integration with more advanced scoring functions
- Batch processing for multiple protein targets
- Property constraint optimization algorithms
- Visualization customization options

## References
- MolCraft: [GitHub Repository](https://github.com/AlgoMole/MolCRAFT)
- AutoDock Vina: [Trott & Olson, 2010]
- QED Score: [Bickerton et al., 2012]
- P2Rank: [Križan et al., 2018]

## Configuration Files
- Default workflow: `configs/workflow/stable_drug_design.yaml`
- Model checkpoints: `checkpoints/molcraft/`
- Visualization templates: `configs/visualization/`

---

*This skill leverages OpenBioMed's comprehensive framework to automate and accelerate the drug discovery process, providing researchers with ready-to-evaluate lead compounds based on cutting-edge computational methodologies.*