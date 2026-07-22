---
name: chembl-query
description: >
  Query ChEMBL database for bioactivity data on drug-like compounds.
  Use this skill when:
  (1) Finding compounds active against a protein target (target-based search),
  (2) Getting bioactivity profile for a molecule (molecule-based search),
  (3) Finding drugs for a disease indication (indication-based search).
license: MIT
category: knowledge-retrieval
tags: [chembl, bioactivity, drug-discovery, target, indication]
---

# ChEMBL Query

Query ChEMBL database for bioactivity data on drug-like compounds, via the OpenBioMed server API.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: if the molecule data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). ChEMBL queries use the endpoint `${OPENBIOMED_API_BASE_URL}/run_pipeline/` with `task: "chembl_query"`.

## When to Use

- Find compounds active against a protein target (target-based search)
- Get bioactivity profile for a molecule (molecule-based search)
- Find drugs for a disease indication (indication-based search)

## Workflow

### Use Case 1: Target-Based Compound Search

Find compounds with activity against a protein target. Set `query_type` to `"target"`.

**Search by target name**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "target", "target_name": "EGFR", "standard_type": "IC50", "standard_value_lte": 100, "limit": 20}'
```

**Search by UniProt ID**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "target", "uniprot_id": "P00533", "standard_type": "IC50", "limit": 20}'
```

Response:
```json
{
  "task": "chembl_query",
  "query_type": "target",
  "results": [
    {
      "molecule_chembl_id": "CHEMBL...",
      "molecule_name": "...",
      "target_chembl_id": "CHEMBL...",
      "target_name": "EGFR",
      "standard_type": "IC50",
      "standard_value": "...",
      "standard_units": "nM",
      "pchembl_value": "..."
    }
  ]
}
```

### Use Case 2: Molecule Bioactivity Profile

Get all known targets and activity data for a compound. Set `query_type` to `"molecule"`.

**Search by molecule name**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "molecule", "molecule_name": "imatinib", "limit": 50}'
```

**Search by SMILES**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "molecule", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "limit": 20}'
```

**Search by ChEMBL ID**:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "molecule", "chembl_id": "CHEMBL25", "limit": 20}'
```

### Use Case 3: Disease/Indication-Based Drug Search

Find drugs studied for a specific disease. Set `query_type` to `"indication"`.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "indication", "disease": "diabetes", "limit": 50}'
```

**Filter for approved drugs only** (max_phase=4):
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "chembl_query", "query_type": "indication", "disease": "diabetes", "max_phase": 4, "limit": 20}'
```

## Expected Outputs

| Query Type | Key Response Fields |
|------------|---------------------|
| target | `molecule_chembl_id`, `molecule_name`, `target_name`, `standard_type`, `standard_value`, `pchembl_value` |
| molecule | `molecule_chembl_id`, `molecule_name`, `target_chembl_id`, `target_name`, `standard_type`, `standard_value`, `pchembl_value` |
| indication | `molecule_chembl_id`, `molecule_name`, `indication`, `max_phase_for_ind`, `phase_description` |

## Parameters Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `query_type` | str (required) | `"target"`, `"molecule"`, or `"indication"` |
| `target_name` | str | Target name (e.g., "EGFR") |
| `uniprot_id` | str | UniProt accession (e.g., "P00533") |
| `molecule_name` | str | Molecule name (e.g., "aspirin") |
| `smiles` | str | SMILES string |
| `chembl_id` | str | ChEMBL ID (e.g., "CHEMBL25") |
| `disease` | str | Disease name (e.g., "diabetes") |
| `standard_type` | str | Activity type (e.g., "IC50", "Ki", "EC50") |
| `standard_value_lte` | int | Max activity value in nM |
| `max_phase` | int | Minimum clinical phase (0-4) |
| `limit` | int | Max results (default: 50) |

## Score Interpretation

### Activity Values (pChEMBL)

| pChEMBL Value | IC50/Ki Approx | Interpretation |
|---------------|----------------|----------------|
| > 9 | < 1 nM | Extremely potent |
| 8-9 | 1-10 nM | Very potent |
| 7-8 | 10-100 nM | Potent |
| 6-7 | 100 nM - 1 uM | Moderately active |
| 5-6 | 1-10 uM | Weakly active |
| < 5 | > 10 uM | Inactive |

### Clinical Phase

| Phase | Description |
|-------|-------------|
| 0 | Preclinical |
| 1 | Phase I (safety) |
| 2 | Phase II (efficacy) |
| 3 | Phase III (large-scale) |
| 4 | Approved |

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Target Not Found

**Symptom**: Empty results list for target search.

**Solution**: Try alternative target names or use `uniprot_id` instead of `target_name`.

### No Activity Data

**Symptom**: Results returned but with no bioactivity values.

**Solution**: The target may not have curated data. Try different `standard_type` values (e.g., "Ki", "EC50").

### Molecule Not Found

**Symptom**: Empty results for molecule search.

**Solution**: Verify SMILES format. Try `molecule_name` or `chembl_id` instead.

### No Indication Results

**Symptom**: Empty results for disease search.

**Solution**: Try simpler disease terms (e.g., "neoplasm" instead of "cancer", "hypertension" instead of specific subtypes).

### Timeout

**Symptom**: curl returns timeout after long wait.

**Solution**: Reduce `limit` parameter (e.g., set `"limit": 10`).

## Notes

- ChEMBL queries are asynchronous — response times may vary depending on data volume
- ChEMBL queries use the `/run_pipeline/` endpoint with `task: "chembl_query"`
- Results are capped by the `limit` parameter — increase it for comprehensive data, decrease for faster responses