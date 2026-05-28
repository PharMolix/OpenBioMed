---
name: binding-affinity-prediction-prodigy
description: >
  Protein complex binding affinity prediction.
  Use this skill when:
  (1) Predict the binding affinity score,
  (2) Using protein complex structure.

license: MIT
category: design-tools
tags: [binding affinity, protein complex, prodigy]
---

# Prodigy Binding Affinity Prediction for Protein Complex

Predict binding affinity for protein-protein complexes using PRODIGY (PROtein binding affinity prediction using contact enerGY).

## When to Use

- User wants to predict binding affinity of a protein complex
- User provides a PDB file containing a protein-protein complex
- User wants to evaluate protein-protein interaction strength

## API Endpoint Resolution

The skill resolves the OpenBioMed API base URL in this order:

1. **Environment variable**: `${OPENBIOMED_API_BASE_URL}` (if set)
2. **Docker container default**: `http://openbiomed-server:8090` (if running in Docker)
3. **Local development default**: `http://127.0.0.1:8090`

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL.

## Workflow

### Step 1: Prepare Protein Complex PDB File

The input must be a PDB file containing a protein-protein complex (two or more interacting proteins).

| Input Type | Example | How to Handle |
|------------|---------|---------------|
| Local PDB file | `./complex.pdb` | Use path directly (must exist on server filesystem) |
| PDB ID | `1AVX` | Download from RCSB PDB first |

**If input is PDB ID**, first download the structure:

```bash
# Option 1: Use OpenBioMed protein_pdb_request
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "protein_pdb_request", "query": "1AVX", "mode": "file_only"}'

# Option 2: Direct download from RCSB PDB
curl -L -o complex.pdb "https://files.rcsb.org/download/1AVX.pdb"
```

### Step 2: Call binding_affinity API

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "binding_affinity", "protein_complex": "<PDB_FILE_PATH>"}'
```

**Optional**: Specify distance cutoff for intermolecular contacts:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "binding_affinity", "protein_complex": "<PDB_FILE_PATH>", "distance_cutoff": 5.5}'
```

**Response**:
```json
{
  "task": "binding_affinity",
  "binding_affinity": -11.6,
  "distance_cutoff": 5.5,
  "description": "Binding affinity: -11.6 kcal.mol-1 (distance_cutoff=5.5)"
}
```

### Step 3: Interpret Results

| Binding Affinity | Interpretation |
|------------------|----------------|
| < -15 kcal/mol | Very strong binding |
| -15 to -10 kcal/mol | Strong binding |
| -10 to -5 kcal/mol | Moderate binding |
| > -5 kcal/mol | Weak binding |

## Example Usage

### Example 1: Predict Binding Affinity

```
Input: "Predict the binding affinity of protein complex 1AVX"

Step 1: Download PDB file
  curl -L -o 1avx.pdb "https://files.rcsb.org/download/1AVX.pdb"
  → File saved: 1avx.pdb

Step 2: Call binding_affinity API

  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "binding_affinity", "protein_complex": "1avx.pdb"}'

Step 3: Interpret results

Output:
  Binding affinity: -11.6 kcal/mol
  Interpretation: Strong binding between proteins in the complex
```

## Expected Outputs

| Output | Type | Description |
|--------|------|-------------|
| binding_affinity | float | Predicted binding affinity in kcal.mol-1 |
| distance_cutoff | float | Distance cutoff used for calculation |
| description | string | Human-readable description |

## Error Handling

### PDB File Not Found

**Symptom**: API returns error about file not found.

**Solution**: Ensure the PDB file path is correct and accessible on the server filesystem.

### Invalid PDB Format

**Symptom**: API returns parsing error or "No contacts found".

**Solution**: Ensure the PDB file is a valid protein-protein complex:
- Contains at least two protein chains
- Chains are close enough to have intermolecular contacts
- File format is standard PDB

## Decision Tree

```
Should I use PRODIGY?
│
└─ What type of complex are you evaluating?
   ├─ Protein-protein complex → binding-affinity-prediction-prodigy ✓
   └─ Protein-ligand complex → Use protein-ligand binding analysis tools
```

## Technical Details

### PRODIGY Algorithm

PRODIGY predicts binding affinity based on:
1. **Intermolecular Contacts (ICs)**: Number of contacts between protein chains
2. **Contact Types**: Classification into different contact categories
3. **Physicochemical Properties**: Incorporation of NIS (Non-Interacting Surface) properties

### Distance Cutoff

The `distance_cutoff` parameter defines the maximum distance (in Angstroms) between atoms to be considered as a contact. Default is 5.5 Å.

| Distance Cutoff | Effect |
|-----------------|--------|
| 5.0 Å | More stringent, fewer contacts counted |
| 5.5 Å | Default, balanced |
| 6.0 Å | More lenient, more contacts counted |

## See Also

- `protein_pdb_request` - Download PDB files from RCSB database
- `analyze_complex_interaction` - Analyze protein-ligand interactions
- `ppi_string_request` - Query STRING database for protein-protein interactions