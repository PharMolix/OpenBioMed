---
name: drug-candidate-discovery
description: >
  Generate diverse druggable molecules for a given target or disease using OpenBioMed's
  AI-powered drug discovery tools via the run_pipeline API. Use this skill when:
  (1) Generating drug candidates, molecules, or compounds for a target/disease,
  (2) Performing structure-based drug design or de novo drug design,
  (3) Finding or creating molecules that bind to a specific protein target,
  (4) Discovering potential drugs for a disease name,
  (5) Designing molecules with specific properties (LogP, QED, docking scores).

  The skill handles target identification, structure retrieval, molecule generation,
  and in silico evaluation through API calls to the OpenBioMed server.
license: MIT
category: drug-discovery
tags: [drug-design, molecule-generation, structure-based-design]
---

# Drug Candidate Discovery

This skill uses the OpenBioMed server API to generate diverse druggable molecules for a given target or disease. It orchestrates a complete drug discovery workflow from target identification to candidate evaluation via run_pipeline endpoint calls.

## Endpoint Configuration (read this first)

Defaults declared in this skill:

- `OPENBIOMED_CLOUD_URL = http://127.0.0.1:8092`
  Placeholder for the OpenBioMed cloud service base URL.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set.
3. Otherwise, ask the user once which endpoint to use, offering these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value.
   - **Self-hosted OpenBioMed server**: user provides their own base URL.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## Inputs

The user should provide:
- **target_or_disease** (required): Name of the target protein or disease (e.g., "BTK", "Alzheimer's disease", "KRAS G12C")
- **num_candidates** (optional, default=5): Number of desired candidate molecules
- **property_constraints** (optional): Desired molecular properties:
  - `logp_min`, `logp_max`: LogP range (e.g., -1 to 3)
  - `qed_min`: Minimum QED score (e.g., 0.5)
  - `sa_min`: Minimum synthetic accessibility score (e.g., 0.5)

## Workflow Overview

### Phase 1: Target Identification & Research

Use web search and database query tools to find target information:

| Step | API Call | Purpose |
|------|----------|---------|
| 1.1 | `web_search` | Search for target protein UniProt ID and drug information |
| 1.2 | `literature_search` | PubMed search for target research papers |
| 1.3 | `chembl_query` | Query ChEMBL for known drugs/inhibitors |

### Phase 2: Structure Retrieval & Preparation

Retrieve and process protein structures:

| Step | API Call | Purpose |
|------|----------|---------|
| 2.1 | `protein_uniprot_request` | Get protein metadata from UniProt |
| 2.2 | `protein_pdb_request` | Download PDB structure with bound ligand |
| 2.3 | `extract_molecules_from_pdb_file` | Extract protein chains and ligand molecules |
| 2.4 | `create_pocket_from_ligand` | Create binding pocket from reference ligand |

### Phase 3: Molecule Generation & Optimization

Generate and evaluate candidate molecules:

| Step | API Call | Purpose |
|------|----------|---------|
| 3.1 | `structure_based_drug_design` | Generate molecules for binding pocket |
| 3.2 | `molecule_property_calculation` | Calculate QED, LogP, SA scores |
| 3.3 | `drug_lead_analysis` | Comprehensive drug-likeness analysis |

### Phase 4: Results & Reporting

Visualize and export results:

| Step | API Call | Purpose |
|------|----------|---------|
| 4.1 | `visualize_molecule` | Generate 2D molecule images |
| 4.2 | `visualize_complex` | Generate protein-ligand complex images |
| 4.3 | `export_molecule` | Export molecules as SDF files |

---

## API Query Types

### Phase 1 APIs

#### web_search
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "web_search", "query": "{target_name} protein UniProt PDB inhibitor"}'
```

#### literature_search (PubMed)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "pubmed_search", "query": "{target_name} inhibitor", "max_results": 10}'
```

#### chembl_query
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "target_search", "target_name": "{target_name}", "limit": 10}'
```

### Phase 2 APIs

#### protein_uniprot_request
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "protein_uniprot_request", "query": "{uniprot_id}"}'
```

Response:
```json
{
  "task": "protein_uniprot_request",
  "protein": "./tmp/protein.pkl",
  "protein_preview": "Protein(name=..., sequence=...)"
}
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
Creates a binding pocket from protein and reference ligand coordinates. **This step is REQUIRED before calling `structure_based_drug_design`.**

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

**Note**: The `similarity` field is used as the radius parameter (default 10.0 Angstroms). The pocket parameter for `structure_based_drug_design` must be a Pocket object created by this API, not a Protein object.

### Phase 3 APIs

#### structure_based_drug_design
Requires a pocket file (binary pickle format). The pocket can be created from protein and ligand coordinates.

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

#### molecule_property_calculation
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_calculation", "molecule": "./tmp/molecule.pkl", "property": "QED"}'
```

Response:
```json
{
  "task": "molecule_property_calculation",
  "score": 0.65
}
```

Available properties: `QED`, `SA`, `LogP`, `Lipinski`

#### drug_lead_analysis
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "drug_lead_analysis", "molecule": "./tmp/molecule.pkl"}'
```

Response:
```json
{
  "task": "drug_lead_analysis",
  "report": {
    "drug_likeness": {"qed": 0.65, "lipinski_violations": 0},
    "admet": {"bbb_penetration": "high", "cyp_inhibition": "low"},
    "safety": {"toxicity_risk": "low"}
  }
}
```

### Phase 4 APIs

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
TARGET_NAME="BACE1"  # Replace with user's target
BASE_URL="${OPENBIOMED_API_BASE_URL}"

# Phase 1: Target Identification
echo "[Phase 1] Searching for target information..."

# Web search
WEB_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"web_search\", \"query\": \"${TARGET_NAME} protein UniProt PDB inhibitor\"}")

# Literature search
LIT_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"literature_search\", \"query_type\": \"pubmed_search\", \"query\": \"${TARGET_NAME} inhibitor\", \"max_results\": 10}")

# ChEMBL query for known drugs
CHEMBL_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"chembl_query\", \"query_type\": \"target_search\", \"target_name\": \"${TARGET_NAME}\"}")

# Extract UniProt ID and PDB ID from web search results
# (Parse JSON to extract identifiers)

# Phase 2: Structure Retrieval
echo "[Phase 2] Retrieving protein structure..."

# Download PDB structure (use a known PDB ID or extracted from search)
PDB_ID="4DJW"  # Example for BACE1
PDB_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"protein_pdb_request\", \"query\": \"${PDB_ID}\"}")

# Extract molecules from PDB
EXTRACT_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"extract_molecules_from_pdb_file\", \"protein\": \"./tmp/pdb_${PDB_ID}.pdb\"}")

# Extract protein and ligand file paths
PROTEIN_FILE=$(echo "$EXTRACT_RESULT" | jq -r '.results[] | select(.type=="protein") | .file')
LIGAND_FILE=$(echo "$EXTRACT_RESULT" | jq -r '.results[] | select(.type=="molecule") | .file')

# Create pocket from ligand (REQUIRED for molecule generation)
POCKET_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"create_pocket_from_ligand\", \"protein\": \"${PROTEIN_FILE}\", \"molecule\": \"${LIGAND_FILE}\", \"similarity\": 10.0}")

POCKET_FILE=$(echo "$POCKET_RESULT" | jq -r '.pocket')

# Phase 3: Molecule Generation
echo "[Phase 3] Generating candidate molecules..."

GENERATED_MOL=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"structure_based_drug_design\", \"model\": \"molcraft\", \"pocket\": \"${POCKET_FILE}\"}")

# Calculate properties
QED_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"molecule_property_calculation\", \"molecule\": \"./tmp/generated_molecule.pkl\", \"property\": \"QED\"}")

LOGP_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"molecule_property_calculation\", \"molecule\": \"./tmp/generated_molecule.pkl\", \"property\": \"LogP\"}")

# Drug lead analysis
LEAD_ANALYSIS=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"drug_lead_analysis\", \"molecule\": \"./tmp/generated_molecule.pkl\"}")

# Phase 4: Visualization and Export
echo "[Phase 4] Generating outputs..."

# Visualize molecule
VISUALIZATION=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"visualize_molecule\", \"visualize\": \"ball_and_stick\", \"molecule\": \"./tmp/generated_molecule.pkl\"}")

# Export molecule as SDF
EXPORT_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"export_molecule\", \"molecule\": \"./tmp/generated_molecule.pkl\"}")

echo "Workflow complete!"
```

---

## Alternative: Simplified Workflow (Scaffold-based)

If the MolCraft model checkpoint is unavailable for structure-based drug design, use a simplified workflow that still leverages the API for property calculation and analysis:

1. Use `web_search` to find target information
2. Use `chembl_query` to retrieve known inhibitor SMILES
3. Use `molecule_property_calculation` to evaluate the retrieved molecules
4. Use `drug_lead_analysis` for comprehensive analysis
5. Use `visualize_molecule` and `export_molecule` for outputs

```bash
# Get known inhibitors from ChEMBL
CHEMBL_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"chembl_query\", \"query_type\": \"target_search\", \"target_name\": \"${TARGET_NAME}\"}")

# For each inhibitor SMILES found:
# - Calculate properties
# - Analyze drug-likeness
# - Export and visualize
```

---

## Expected Outputs

| Output | Description | API Response Field |
|--------|-------------|-------------------|
| Protein structure | Downloaded PDB file | `protein` from `protein_pdb_request` |
| Ligand molecule | Extracted ligand from PDB | `results[].file` from `extract_molecules_from_pdb_file` |
| Binding pocket | Pocket centered on reference ligand | `pocket` from `create_pocket_from_ligand` |
| Generated molecules | New candidate molecules | `molecule` from `structure_based_drug_design` |
| Property scores | QED, LogP, SA | `score` from `molecule_property_calculation` |
| Drug analysis report | Comprehensive lead analysis | `report` from `drug_lead_analysis` |
| Visualization images | 2D/3D molecule images | `image` from `visualize_molecule` |
| SDF files | Molecular structure files | `file` from `export_molecule` |

---

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify endpoint health: `curl ${OPENBIOMED_API_BASE_URL}/healthz`. Re-resolve base URL if needed.

### PDB Structure Not Found

**Symptom**: `protein_pdb_request` returns error or empty file.

**Solution**: Try alternative PDB IDs from web search results. Use RCSB PDB search API directly.

### Molecule Generation Failed

**Symptom**: `structure_based_drug_design` returns error: `'Protein' object has no attribute 'atoms'`.

**Cause**: The `pocket` parameter was passed a Protein file instead of a Pocket file.

**Solution**: Call `create_pocket_from_ligand` first to create a proper pocket file:
```bash
# Step 1: Extract protein and ligand from PDB
# Step 2: Create pocket from ligand
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -d '{"task": "create_pocket_from_ligand", "protein": "./tmp/protein_A.pkl", "molecule": "./tmp/ligand.pkl"}'
# Step 3: Use the returned pocket file for molecule generation
```

### Pocket Creation Failed

**Symptom**: `create_pocket_from_ligand` returns error.

**Solution**: Ensure both protein and ligand files exist and are valid pickle files extracted from `extract_molecules_from_pdb_file`.

### No Ligands Extracted

**Symptom**: `extract_molecules_from_pdb_file` returns no molecules.

**Solution**: The PDB structure may not contain small molecule ligands. Try another structure with co-crystallized inhibitor.

### Molecule Generation Failed

**Symptom**: `structure_based_drug_design` returns error.

**Solution**: Ensure pocket file is valid. Check model checkpoint availability. Use scaffold-based alternative workflow.

---

## Limitations

- `structure_based_drug_design` requires a valid Pocket file created by `create_pocket_from_ligand` (not a Protein file)
- MolCraft model checkpoint must be available at `./checkpoints/molcraft/`
- KEGG/PubChem coverage limited to known molecules
- API rate limits apply for external database queries

## Example Usage

**Input**: "Generate drug candidates for BACE1 for Alzheimer's disease therapy"

**Workflow**:
1. Search for BACE1 UniProt ID (P56817) and PDB structures
2. Download PDB structure with inhibitor (e.g., 4DJW)
3. Extract protein and ligand molecules
4. Create binding pocket from reference ligand
5. Generate new molecules for the binding pocket
6. Calculate drug-likeness properties
7. Analyze and export top candidates

**Input**: "Find potential KRAS G12C inhibitors"

**Workflow**:
1. Search for KRAS G12C target information
2. Query ChEMBL for existing inhibitors
3. Retrieve and evaluate candidate molecules
4. Perform drug lead analysis
5. Generate comprehensive report