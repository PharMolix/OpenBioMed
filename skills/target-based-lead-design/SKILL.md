---
name: target-based-lead-design
description: >
  Generate diverse lead compounds for a specific protein target using structure-based
  drug design with MolCraft. Use this skill when:
  (1) Designing drug candidates for a known protein target (PDB ID or disease name),
  (2) Generating structurally diverse molecules with optimized binding affinity,
  (3) Filtering candidates based on user-defined criteria (docking, ADMET, drug-likeness),
  (4) Iteratively refining leads through regeneration when criteria are not met.

  The skill handles target identification, structure retrieval, molecule generation,
  docking, property calculation, and in silico evaluation through API calls.
license: MIT
category: drug-discovery
tags: [lead-generation, structure-based-design, diversity, molcraft, docking, admet]
---

# Target-Based Lead Design

Generate diverse, drug-like lead compounds targeting a specific protein using AI-powered structure-based drug design via the run_pipeline API.

## Endpoint Configuration (read this first)

Defaults declared in this skill:

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  Placeholder for the OpenBioMed cloud service base URL.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set.
3. Otherwise, ask the user once which endpoint to use, offering these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value.
   - **Self-hosted OpenBioMed server**: user provides their own base URL.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | str | Yes | PDB ID (e.g., "4xli") or disease/target name |
| `num_candidates` | int | No | Initial candidates to generate (default: 40) |
| `target_leads` | int | No | Desired number of final leads (default: 20) |

### User Criteria (Filtering Thresholds)

| Criterion | Default | Description |
|-----------|---------|-------------|
| `docking_threshold` | -10.0 | Maximum docking score (kcal/mol), more negative = better |
| `qed_min` | 0.4 | Minimum QED score (0-1), higher = more drug-like |
| `lipinski_min` | 4 | Minimum Lipinski rules obeyed (0-4), 4 = no violations |
| `similarity_max` | 0.7 | Maximum Tanimoto similarity between selected leads |

## Workflow Overview

### Phase 1: Target Identification & Structure Retrieval

| Step | API Call | Purpose |
|------|----------|---------|
| 1.1 | `web_search` | Search for target PDB structures (if disease name provided) |
| 1.2 | `protein_pdb_request` | Download PDB structure file |

### Phase 2: Structure Preparation

| Step | API Call | Purpose |
|------|----------|---------|
| 2.1 | `extract_molecules_from_pdb_file` | Extract protein chains and ligands from PDB |
| 2.2 | `create_pocket_from_ligand` | Create binding pocket from reference ligand |

### Phase 3: Molecule Generation

| Step | API Call | Purpose |
|------|----------|---------|
| 3.1 | `structure_based_drug_design` | Generate molecules for binding pocket |

### Phase 4: Docking

| Step | API Call | Purpose |
|------|----------|---------|
| 4.1 | `protein_molecule_docking_score` | Dock candidates against protein |

### Phase 5: Property Calculation

| Step | API Call | Purpose |
|------|----------|---------|
| 5.1 | `molecule_property_calculation` | Calculate QED, SA, LogP, Lipinski |

### Phase 6: Filtering & Diversity

| Step | API Call | Purpose |
|------|----------|---------|
| 6.1 | `molecule_similarity` | Calculate pairwise Tanimoto similarity |

### Phase 7: Interaction Analysis

| Step | API Call | Purpose |
|------|----------|---------|
| 7.1 | `analyze_complex_interaction` | Analyze protein-ligand interactions (PLIP) |

### Phase 8: Visualization

| Step | API Call | Purpose |
|------|----------|---------|
| 8.1 | `visualize_molecule` | Generate 2D molecule images |
| 8.2 | `visualize_complex` | Generate protein-ligand complex images |
| 8.3 | `export_molecule` | Export molecules as SDF files |

---

## API Query Types

### Phase 1 APIs

#### web_search
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "web_search", "query": "{target_name} PDB structure inhibitor"}'
```

#### protein_pdb_request
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "protein_pdb_request", "query": "{pdb_id}"}'
```

Response:
```json
{
  "task": "protein_pdb_request",
  "protein": "./tmp/pdb_{pdb_id}.pdb",
  "protein_preview": "Protein(name={pdb_id}, ...)"
}
```

### Phase 2 APIs

#### extract_molecules_from_pdb_file
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "extract_molecules_from_pdb_file", "protein": "./tmp/pdb_{pdb_id}.pdb"}'
```

Response:
```json
{
  "task": "extract_molecules_from_pdb_file",
  "results": [
    {"type": "protein", "chain_id": "A", "name": "{pdb_id}_A", "sequence_preview": "...", "file": "./tmp/{pdb_id}_A.pkl"},
    {"type": "molecule", "chain_id": "A", "name": "...", "smiles": "...", "file": "./tmp/{molecule_name}.pkl"}
  ],
  "metadata": "Total 1 protein chains, 1 molecules and 0 ions extracted..."
}
```

#### create_pocket_from_ligand
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "create_pocket_from_ligand", "protein": "./tmp/{pdb_id}_A.pkl", "molecule": "./tmp/{ligand_name}.pkl", "similarity": 10.0}'
```

Response:
```json
{
  "task": "create_pocket_from_ligand",
  "pocket": "./tmp/pocket.pkl",
  "pocket_preview": "Pocket(...)"
}
```

Note: The `similarity` field is used as the radius parameter (default 10.0 Angstroms).

### Phase 3 APIs

#### structure_based_drug_design
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "structure_based_drug_design", "model": "molcraft", "pocket": "./tmp/pocket.pkl"}'
```

Response:
```json
{
  "task": "structure_based_drug_design",
  "model": "molcraft",
  "molecule": "./tmp/generated_molecule.pkl",
  "molecule_preview": "CC1=CC=CC=C1..."
}
```

### Phase 4 APIs

#### protein_molecule_docking_score
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "protein_molecule_docking_score", "protein": "./tmp/{pdb_id}_A.pkl", "molecule": "./tmp/generated_molecule.pkl"}'
```

Response:
```json
{
  "task": "protein_molecule_docking_score",
  "score": "-9.5"
}
```

### Phase 5 APIs

#### molecule_property_calculation
```bash
# Calculate QED
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_calculation", "molecule": "./tmp/molecule.pkl", "property": "QED"}'

# Calculate LogP
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_calculation", "molecule": "./tmp/molecule.pkl", "property": "LogP"}'

# Calculate Lipinski
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_calculation", "molecule": "./tmp/molecule.pkl", "property": "Lipinski"}'
```

Response:
```json
{
  "task": "molecule_property_calculation",
  "score": 0.65
}
```

Available properties: `QED`, `SA`, `LogP`, `Lipinski`

### Phase 6 APIs

#### molecule_similarity
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_similarity", "molecule_1": "./tmp/mol1.pkl", "molecule_2": "./tmp/mol2.pkl"}'
```

Response:
```json
{
  "task": "molecule_similarity",
  "similarity": 0.42
}
```

### Phase 7 APIs

#### analyze_complex_interaction
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "analyze_complex_interaction", "molecule": "./tmp/molecule.pkl", "protein": "./tmp/protein.pkl"}'
```

Response:
```json
{
  "task": "analyze_complex_interaction",
  "report": "UNL:hydrophobic interactions;H-bonds;..."
}
```

### Phase 8 APIs

#### visualize_molecule
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "visualize_molecule", "visualize": "ball_and_stick", "molecule": "./tmp/molecule.pkl"}'
```

Response:
```json
{
  "task": "visualize_molecule",
  "image": "https://.../molecule_visualization.png"
}
```

#### visualize_complex
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "visualize_complex", "protein": "./tmp/protein.pkl", "molecule": "./tmp/molecule.pkl"}'
```

#### export_molecule
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "export_molecule", "molecule": "./tmp/molecule.pkl"}'
```

Response:
```json
{
  "task": "export_molecule",
  "file": "./tmp/molecule.sdf"
}
```

---

## Complete Workflow Script

### Step 1: Determine Endpoint

First resolve the API base URL per the resolution order above.

### Step 2: Execute Workflow via API Calls

**IMPORTANT: Execute each API call sequentially and save the outputs.**

```bash
# Configuration
TARGET_PDB="4xli"  # Replace with user's target PDB ID
NUM_CANDIDATES=40
TARGET_LEADS=20
BASE_URL="${OPENBIOMED_API_BASE_URL}"

# Phase 1: Structure Retrieval
echo "[Phase 1] Retrieving protein structure..."

PDB_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"protein_pdb_request\", \"query\": \"${TARGET_PDB}\"}")

# Extract PDB file path from response
PDB_FILE=$(echo "$PDB_RESULT" | jq -r '.protein')

# Phase 2: Structure Preparation
echo "[Phase 2] Extracting molecules and creating pocket..."

# Extract molecules from PDB
EXTRACT_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"extract_molecules_from_pdb_file\", \"protein\": \"${PDB_FILE}\"}")

# Extract protein and ligand file paths
PROTEIN_FILE=$(echo "$EXTRACT_RESULT" | jq -r '.results[] | select(.type=="protein") | .file')
LIGAND_FILE=$(echo "$EXTRACT_RESULT" | jq -r '.results[] | select(.type=="molecule") | .file')

# Create pocket from ligand
POCKET_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"create_pocket_from_ligand\", \"protein\": \"${PROTEIN_FILE}\", \"molecule\": \"${LIGAND_FILE}\", \"similarity\": 10.0}")

POCKET_FILE=$(echo "$POCKET_RESULT" | jq -r '.pocket')

# Phase 3: Molecule Generation
echo "[Phase 3] Generating candidate molecules..."

for i in $(seq 1 $NUM_CANDIDATES); do
  echo "Generating molecule $i..."
  GENERATED_MOL=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"structure_based_drug_design\", \"model\": \"molcraft\", \"pocket\": \"${POCKET_FILE}\"}")

  MOL_FILE=$(echo "$GENERATED_MOL" | jq -r '.molecule')

  # Phase 4: Docking
  DOCKING_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"protein_molecule_docking_score\", \"protein\": \"${PROTEIN_FILE}\", \"molecule\": \"${MOL_FILE}\"}")

  DOCKING_SCORE=$(echo "$DOCKING_RESULT" | jq -r '.score')

  # Phase 5: Property Calculation
  QED_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"molecule_property_calculation\", \"molecule\": \"${MOL_FILE}\", \"property\": \"QED\"}")

  LOGP_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"molecule_property_calculation\", \"molecule\": \"${MOL_FILE}\", \"property\": \"LogP\"}")

  LIPINSKI_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"molecule_property_calculation\", \"molecule\": \"${MOL_FILE}\", \"property\": \"Lipinski\"}")

  # Store results for filtering
  echo "$i,$MOL_FILE,$DOCKING_SCORE,$(echo "$QED_RESULT" | jq -r '.score'),$(echo "$LIPINSKI_RESULT" | jq -r '.score')" >> candidates.csv
done

# Phase 6: Filtering & Diversity Selection
echo "[Phase 6] Filtering candidates by criteria..."

# Apply user criteria (bash filtering)
awk -F, -v dock_thresh=-10 -v qed_min=0.4 -v lipinski_min=4 \
  '$3 <= dock_thresh && $4 >= qed_min && $5 >= lipinski_min {print}' candidates.csv > filtered.csv

# Calculate pairwise similarity for diversity selection
# (This requires iterating over filtered candidates and computing molecule_similarity)

# Phase 7: Interaction Analysis for selected leads
echo "[Phase 7] Analyzing protein-ligand interactions..."

# For each selected molecule:
SELECTED_MOL="./tmp/selected_molecule.pkl"
INTERACTION_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"analyze_complex_interaction\", \"molecule\": \"${SELECTED_MOL}\", \"protein\": \"${PROTEIN_FILE}\"}")

# Phase 8: Visualization
echo "[Phase 8] Generating visualizations..."

# Visualize molecule
VISUALIZATION=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"visualize_molecule\", \"visualize\": \"ball_and_stick\", \"molecule\": \"${SELECTED_MOL}\"}")

# Export molecule as SDF
EXPORT_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"export_molecule\", \"molecule\": \"${SELECTED_MOL}\"}")

echo "Workflow complete!"
```

---

## Expected Outputs

| Output | Description | API Response Field |
|--------|-------------|-------------------|
| Protein structure | Downloaded PDB file | `protein` from `protein_pdb_request` |
| Ligand molecule | Extracted ligand from PDB | `results[].file` from `extract_molecules_from_pdb_file` |
| Binding pocket | Pocket centered on reference ligand | `pocket` from `create_pocket_from_ligand` |
| Generated molecules | New candidate molecules | `molecule` from `structure_based_drug_design` |
| Docking scores | Binding affinity scores | `score` from `protein_molecule_docking_score` |
| Property scores | QED, LogP, Lipinski | `score` from `molecule_property_calculation` |
| Interaction reports | PLIP analysis for selected leads | `report` from `analyze_complex_interaction` |
| Visualization images | 2D/3D molecule images | `image` from `visualize_molecule` |
| SDF files | Molecular structure files | `file` from `export_molecule` |

---

## Output Interpretation

### Docking Score (kcal/mol)
| Score | Assessment |
|-------|------------|
| < -10 | Excellent binding |
| -10 to -7 | Good binding |
| -7 to -5 | Moderate binding |
| > -5 | Weak binding |

### QED (Quantitative Estimate of Drug-likeness)
| Score | Assessment |
|-------|------------|
| > 0.7 | Excellent drug-likeness |
| 0.5 - 0.7 | Good drug-likeness |
| 0.4 - 0.5 | Acceptable |
| < 0.4 | Poor drug-likeness |

### Lipinski Rules Obeyed
| Count | Violations | Assessment |
|-------|------------|------------|
| 4 | 0 | Perfect compliance |
| 3 | 1 | Acceptable |
| 2 | 2 | Marginal |
| < 2 | > 2 | May have issues |

---

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify endpoint health: `curl ${OPENBIOMED_API_BASE_URL}/healthz`. Re-resolve base URL if needed.

### PDB Structure Not Found

**Symptom**: `protein_pdb_request` returns error or empty file.

**Solution**: Check PDB ID validity. Use web search to find alternative structures.

### No Ligands Extracted

**Symptom**: `extract_molecules_from_pdb_file` returns no molecules.

**Solution**: The PDB structure may not contain small molecule ligands. Try another structure with co-crystallized inhibitor. If no ligands, use `import_pocket` with manually specified residue indices.

### Pocket Creation Failed

**Symptom**: `create_pocket_from_ligand` returns error.

**Solution**: Ensure both protein and ligand files exist and are valid pickle files.

### Molecule Generation Failed

**Symptom**: `structure_based_drug_design` returns error.

**Solution**: Ensure pocket file is valid. Check MolCraft model checkpoint availability at `./checkpoints/molcraft/`.

### No Candidates Pass Criteria

**Symptom**: Filtering returns empty set.

**Solution**: Relax criteria (increase docking_threshold, lower qed_min) or generate more candidates.

---

## Limitations

- `structure_based_drug_design` requires valid pocket file (created from protein-ligand coordinates)
- MolCraft model checkpoint must be available at `./checkpoints/molcraft/`
- Multiple molecule generation calls are needed for diversity (one per seed)
- API rate limits apply for external database queries
- PLIP (`analyze_complex_interaction`) requires plip package installation on server

## Example Usage

**Input**: "Generate leads for 4xli (ABL2 kinase) with docking threshold -10, QED minimum 0.4"

**Workflow**:
1. Download PDB structure 4xli
2. Extract protein chain and reference ligand
3. Create binding pocket from ligand
4. Generate 40 candidate molecules
5. Dock each candidate
6. Calculate QED, LogP, Lipinski for each
7. Filter by docking <= -10, QED >= 0.4, Lipinski >= 4
8. Select diverse leads (similarity <= 0.7)
9. Analyze interactions with PLIP
10. Visualize and export top leads

**Input**: "Find drug candidates for BACE1 for Alzheimer's disease"

**Workflow**:
1. Web search for BACE1 PDB structures
2. Select best PDB with inhibitor (e.g., 4DJW)
3. Extract protein and ligand
4. Create pocket and generate candidates
5. Filter and select diverse leads