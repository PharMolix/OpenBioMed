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

| Input Type | How to Handle |
|------------|---------------|
| **Uploaded file** | Use file_id directly in http_request (see below) |
| PDB ID | Download from RCSB PDB first (see below) |

#### Uploading User Files

When the user has uploaded a file, you will see a file_id (UUID format) in the conversation. Use the `http_request` tool with the `files` parameter to upload it to the server:

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<file_id>"}'
```

The system will automatically:
- Resolve the file_id to the actual file on disk
- Read the file bytes and send as multipart/form-data
- Inject the required API Key header

The response will contain the server path: `{"path": "./tmp/uploads/<uuid>.pdb"}`

Use this `path` value as the `protein_complex` parameter in Step 2.

#### If input is PDB ID, first download the structure:

```
url: "${OPENBIOMED_API_BASE_URL}/web_search/"
method: "POST"
headers: '{"Content-Type": "application/json"}'
body: '{"task": "protein_pdb_request", "query": "1AVX", "mode": "file_only"}'
```

### Step 2: Call binding_affinity API

```
url: "${OPENBIOMED_API_BASE_URL}/run_pipeline/"
method: "POST"
headers: '{"Content-Type": "application/json"}'
body: '{"task": "binding_affinity", "protein_complex": "<PDB_FILE_PATH>"}'
```

**Optional**: Specify distance cutoff for intermolecular contacts:

```
url: "${OPENBIOMED_API_BASE_URL}/run_pipeline/"
method: "POST"
headers: '{"Content-Type": "application/json"}'
body: '{"task": "binding_affinity", "protein_complex": "<PDB_FILE_PATH>", "distance_cutoff": 5.5}'
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

### Example 1: Predict Binding Affinity (with PDB ID)

```
Input: "Predict the binding affinity of protein complex 1AVX"

Step 1: Download PDB file via http_request
  url: "${OPENBIOMED_API_BASE_URL}/web_search/"
  method: "POST"
  headers: '{"Content-Type": "application/json"}'
  body: '{"task": "protein_pdb_request", "query": "1AVX", "mode": "file_only"}'
  → Response path: "./tmp/pdb_1AVX.pkl"

Step 2: Call binding_affinity API via http_request
  url: "${OPENBIOMED_API_BASE_URL}/run_pipeline/"
  method: "POST"
  headers: '{"Content-Type": "application/json"}'
  body: '{"task": "binding_affinity", "protein_complex": "./tmp/pdb_1AVX.pkl"}'

Step 3: Interpret results
  Binding affinity: -11.6 kcal/mol → Strong binding
```

### Example 2: Predict Binding Affinity (with uploaded file)

```
Input: "Predict the binding affinity of my uploaded complex file"
  (file_id appears in conversation: e.g. cf2e819d-2858-4adc-a176-464d55352c0a)

Step 1: Upload file to server via http_request
  url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
  method: "POST"
  files: '{"file": "cf2e819d-2858-4adc-a176-464d55352c0a"}'
  → Response: {"path": "./tmp/uploads/abc123.pdb"}

Step 2: Call binding_affinity API via http_request
  url: "${OPENBIOMED_API_BASE_URL}/run_pipeline/"
  method: "POST"
  headers: '{"Content-Type": "application/json"}'
  body: '{"task": "binding_affinity", "protein_complex": "./tmp/uploads/abc123.pdb"}'

Step 3: Interpret results
  Binding affinity: -12.5 kcal/mol → Strong binding
```

## Expected Outputs

| Output | Type | Description |
|--------|------|-------------|
| binding_affinity | float | Predicted binding affinity in kcal.mol-1 |
| distance_cutoff | float | Distance cutoff used for calculation |
| description | string | Human-readable description |

## Error Handling

### PDB File Not Found
- **Symptom**: API returns `{"binding_affinity": 0.0, "description": "Error: PDB file not found"}`
- **Solution**: Re-upload the file first, then use the returned path

### Invalid PDB Format
- **Symptom**: API returns `{"binding_affinity": 0.0, "description": "Error: ..."}`
- **Solution**: Ensure the PDB file is a valid protein-protein complex containing at least two interacting protein chains

### Upload Failed
- **Symptom**: Upload returns error status code (4xx/5xx)
- **Solution**: Retry the upload. The system handles multipart encoding and API key automatically

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
