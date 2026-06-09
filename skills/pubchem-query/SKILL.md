---
name: pubchem-query
description: >
  Query PubChem database for chemical structures, similar compounds, and bioactivity data.
  Use this skill when:
  (1) Converting drug name to molecular structure (SMILES, SDF),
  (2) Finding similar compounds for lead optimization,
  (3) Querying bioactivity data against protein targets,
  (4) Getting compounds active in specific assays.

  The skill handles name-to-structure conversion, similarity search, and bioactivity
  queries through API calls to the OpenBioMed server.
license: MIT
category: data-retrieval
tags: [pubchem, compound-search, bioactivity, similarity-search]
---

# PubChem Query

Query PubChem database for drug discovery and chemistry applications via the OpenBioMed server API.

## Endpoint Configuration

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  The OpenBioMed cloud service base URL.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash).

## Inputs

| Query Type | Required Parameters | Optional Parameters |
|------------|---------------------|---------------------|
| Name to Structure | `query` (drug name or CID) | - |
| Similarity Search | `molecule` (SMILES/file), `threshold` | `max_records` |
| Bioactivity (compound) | `query_type="compound"` | `cid`, `aids_type` |
| Bioactivity (assay) | `query_type="assay"` | `aid`, `cids_type` |
| Bioactivity (target) | `query_type="target"` | `gene_symbol` or `gene_id` |

---

## API Query Types

### 1. Name/ID to Structure (`molecule_name_request`)

Convert drug name or PubChem CID to molecular structure.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}'
```

Response:
```json
{
  "task": "molecule_name_request",
  "molecule": "./tmp/pubchem_aspirin.pkl",
  "molecule_preview": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

Query by PubChem CID:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "2244"}'
```

### 2. Similarity Search (`molecule_structure_request`)

Find similar compounds based on structure.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_structure_request", "molecule": "CC(=O)OC1=CC=CC=C1C(=O)O", "threshold": "0.85"}'
```

Response:
```json
{
  "task": "molecule_structure_request",
  "molecule": "./tmp/similar_compound.pkl",
  "molecule_preview": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

**Note**: The molecule parameter can be:
- SMILES string (e.g., `"CC(=O)OC1=CC=CC=C1C(=O)O"`)
- SDF file path (e.g., `"./tmp/molecule.sdf"`)
- Pickle file path (e.g., `"./tmp/molecule.pkl"`)
- File paths must be accessible on the server filesystem

### 3. Bioactivity Queries (`pubchem_bioactivity`)

#### Query by Target Gene

Get assays targeting a specific gene.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "pubchem_bioactivity", "query_type": "target", "gene_symbol": "HMGCR"}'
```

Response:
```json
{
  "task": "pubchem_bioactivity",
  "query_type": "target",
  "results": [
    {"AID": 1053202, "type": "assay_id"},
    {"AID": 1234567, "type": "assay_id"}
  ]
}
```

#### Query by Compound CID

Get assays where a compound was tested.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "pubchem_bioactivity", "query_type": "compound", "cid": 2244, "aids_type": "active"}'
```

Response:
```json
{
  "task": "pubchem_bioactivity",
  "query_type": "compound",
  "results": [
    {"AID": 1053202, "type": "assay_id"},
    {"AID": 1234567, "type": "assay_id"}
  ]
}
```

`aids_type` options:
- `"active"` - Assays where compound was active
- `"inactive"` - Assays where compound was inactive

#### Query by Assay AID

Get compounds active/inactive in a specific assay.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "pubchem_bioactivity", "query_type": "assay", "aid": 1053202, "cids_type": "active"}'
```

Response:
```json
{
  "task": "pubchem_bioactivity",
  "query_type": "assay",
  "results": [
    {"CID": 2244, "type": "compound_id"},
    {"CID": 5678, "type": "compound_id"}
  ]
}
```

`cids_type` options:
- `"active"` - Compounds that were active
- `"inactive"` - Compounds that were inactive

---

## Complete Workflow Examples

### Example 1: Convert Drug Name to SMILES

```bash
BASE_URL="${OPENBIOMED_API_BASE_URL}"

# Query aspirin
RESULT=$(curl -s -X POST "${BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}')

SMILES=$(echo "$RESULT" | jq -r '.molecule_preview')
MOL_FILE=$(echo "$RESULT" | jq -r '.molecule')

echo "Aspirin SMILES: $SMILES"
echo "Molecule file: $MOL_FILE"
```

### Example 2: Find Similar Compounds

```bash
BASE_URL="${OPENBIOMED_API_BASE_URL}"
QUERY_SMILES="CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin

# Find compounds with >85% similarity
RESULT=$(curl -s -X POST "${BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"molecule_structure_request\", \"molecule\": \"${QUERY_SMILES}\", \"threshold\": \"0.85\"}")

echo "$RESULT" | jq '.molecule_preview'
```

### Example 3: Get Active Compounds for a Target

```bash
BASE_URL="${OPENBIOMED_API_BASE_URL}"
TARGET_GENE="PTGS2"  # COX-2

# Step 1: Get assays targeting PTGS2
ASSAYS=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"pubchem_bioactivity\", \"query_type\": \"target\", \"gene_symbol\": \"${TARGET_GENE}\"}")

# Step 2: Get active compounds from first assay
FIRST_AID=$(echo "$ASSAYS" | jq -r '.results[0].AID')

COMPOUNDS=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"pubchem_bioactivity\", \"query_type\": \"assay\", \"aid\": ${FIRST_AID}, \"cids_type\": \"active\"}")

echo "Active compounds for assay $FIRST_AID:"
echo "$COMPOUNDS" | jq '.results'
```

---

## Expected Outputs

| Query Type | Output Fields |
|------------|---------------|
| Name to Structure | `molecule` (file path), `molecule_preview` (SMILES) |
| Similarity Search | `molecule` (similar compound file), `molecule_preview` |
| Bioactivity (target) | `results` - list of assay IDs |
| Bioactivity (compound) | `results` - list of assay IDs |
| Bioactivity (assay) | `results` - list of compound IDs |

---

## Reading Response Data

**After receiving the response:**
- For `molecule_name_request` and `molecule_structure_request`: Use `molecule_preview` field for the SMILES string. The `molecule` field is a `.pkl` binary file — note its path for downstream use (e.g., as input to other APIs like `pocket_molecule_docking`), but it cannot be directly read as text.
- For `pubchem_bioactivity`: The `results` field contains structured data (AID/CID lists) — parse directly from the JSON response.

---

## Score Interpretation

| Similarity Threshold | Interpretation |
|---------------------|----------------|
| > 0.90 | Very similar, likely same scaffold |
| 0.80-0.90 | Similar, potential analogs |
| 0.70-0.80 | Moderately similar, scaffold hops possible |

---

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify endpoint health: `curl ${OPENBIOMED_API_BASE_URL}/healthz`. Re-resolve base URL if needed.

### Compound Not Found

**Symptom**: `molecule_name_request` returns error.

**Solution**: Try alternative names (brand name, generic name) or use PubChem CID directly.

### No Similar Compounds Found

**Symptom**: `molecule_structure_request` returns empty or error.

**Solution**: Lower threshold (minimum 0.70). Try different query molecule.

### No Bioactivity Data

**Symptom**: `pubchem_bioactivity` returns empty results.

**Solution**: The compound/target may not be tested. Try related compounds or similar targets.

### Timeout

**Symptom**: Request takes too long.

**Solution**: Reduce `max_records` parameter. Retry after a few seconds.

---

## Limitations

- PubChem API has rate limits (5 requests per second by default)
- Similarity search returns one random similar compound (not full list)
- Bioactivity data depends on what PubChem has indexed
- Large result sets may be truncated

## Related Skills

- **molecule-property-prediction**: Predict molecular properties (QED, LogP, etc.) for compounds retrieved from PubChem
- **protein-binding-site-prediction**: Predict binding sites on proteins identified via bioactivity queries
- **pocket-molecule-docking**: Dock PubChem-retrieved compounds into protein pockets
- **drug-lead-analysis**: Comprehensive drug lead analysis for retrieved compounds

## Example Usage

**Input**: "Get the structure of ibuprofen"

**Workflow**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -d '{"task": "molecule_name_request", "query": "ibuprofen"}'
```

**Input**: "Find compounds similar to aspirin with >80% similarity"

**Workflow**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -d '{"task": "molecule_structure_request", "molecule": "CC(=O)OC1=CC=CC=C1C(=O)O", "threshold": "0.80"}'
```

**Input**: "What assays target HMGCR (cholesterol synthesis)?"

**Workflow**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -d '{"task": "pubchem_bioactivity", "query_type": "target", "gene_symbol": "HMGCR"}'
```