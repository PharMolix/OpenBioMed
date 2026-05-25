---
name: drug-drug-interaction-analysis
description: Analyze potential drug-drug interactions (DDI) for up to 5 drugs using KEGG DDI database. Use this skill when: (1) Checking interactions between multiple medications, (2) Assessing DDI risk for drug combinations, (3) Understanding interaction mechanisms and severity, (4) Analyzing CYP enzyme involvement in DDIs.
license: MIT
---

# Drug-Drug Interaction Analysis

Analyze potential drug-drug interactions (DDI) for medication safety assessment via the OpenBioMed server API.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://127.0.0.1:8092`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `http://localhost:9000` or `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: if the drug data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## When to Use

- Checking interactions between prescribed medications
- Evaluating DDI risk for polypharmacy patients
- Understanding interaction mechanisms (CYP enzymes, shared targets)
- Clinical decision support for drug combinations

## API Query Types

The `ddi_analysis` task supports the following query types via `query_type` parameter:

| query_type | Description | Required Params | Optional Params |
|------------|-------------|-----------------|-----------------|
| `find_drug` | Find KEGG drug ID from name | `query` | - |
| `get_drug_info` | Get detailed drug information | `drug_id` | - |
| `get_interactions` | Query DDI for drug IDs | `drug_ids` | - |
| `analyze` | Complete DDI analysis workflow | `drugs` | - |

Note: `drugs` can be a comma-separated string or a list. Maximum 5 drugs per analysis.

## Workflow

### Step 1: Analyze Drug Interactions (Recommended)

For complete DDI analysis, use the `analyze` query type:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ddi_analysis", "query_type": "analyze", "drugs": ["aspirin", "warfarin", "omeprazole"]}'
```

Or with comma-separated string:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ddi_analysis", "query_type": "analyze", "drugs": "aspirin,warfarin,omeprazole"}'
```

Response:
```json
{
  "task": "ddi_analysis",
  "query_type": "analyze",
  "results": [{
    "drugs_analyzed": ["aspirin", "warfarin", "omeprazole"],
    "drug_ids": {"aspirin": "D00109", "warfarin": "D00486", "omeprazole": "D00456"},
    "total_pairs": 3,
    "interactions_found": 2,
    "severity_summary": {"Contraindicated": 1, "Precaution": 1, "Caution": 0},
    "interactions": [
      {
        "drug_a": "aspirin (D00109)",
        "drug_b": "warfarin (D00486)",
        "severity": "Contraindicated",
        "severity_code": "CI",
        "mechanism": "Aspirin may increase anticoagulant effect of warfarin"
      }
    ],
    "drug_details": {...},
    "unresolved": []
  }]
}
```

### Step 2: Find Drug ID (Optional)

If you need to resolve a drug name to KEGG ID:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ddi_analysis", "query_type": "find_drug", "query": "ibuprofen"}'
```

Response:
```json
{
  "task": "ddi_analysis",
  "query_type": "find_drug",
  "results": [{
    "drug_name": "ibuprofen",
    "kegg_id": "D00126",
    "matched_name": "Ibuprofen"
  }]
}
```

### Step 3: Get Drug Info (Optional)

For detailed drug information including targets and metabolism:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ddi_analysis", "query_type": "get_drug_info", "drug_id": "D00109"}'
```

Response:
```json
{
  "task": "ddi_analysis",
  "query_type": "get_drug_info",
  "results": [{
    "kegg_id": "D00109",
    "name": "Aspirin",
    "formula": "C9H8O4",
    "targets": ["PTGS1", "PTGS2"],
    "enzymes": ["CYP2C9", "CYP2C19"],
    "atc_codes": ["B01AC06", "N02BA01"]
  }]
}
```

### Step 4: Query Interactions by IDs (Optional)

For direct DDI query using known KEGG IDs:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ddi_analysis", "query_type": "get_interactions", "drug_ids": ["D00109", "D00126"]}'
```

## Expected Outputs

| Query Type | Response Field | Output |
|------------|---------------|--------|
| `find_drug` | `results` | KEGG ID and matched name |
| `get_drug_info` | `results` | Drug details (name, formula, targets, enzymes) |
| `get_interactions` | `results` | List of interactions between drugs |
| `analyze` | `results` | Complete analysis with interactions, severity summary |

### Severity Levels

| Code | Severity | Description |
|------|----------|-------------|
| **CI** | Contraindicated | Should not be used together |
| **P** | Precaution | Monitor closely; adjust if needed |
| **C** | Caution | Be aware; may need intervention |

### Mechanism Types

- **Target overlap**: Both drugs act on same protein targets
- **CYP interaction**: One drug affects metabolism of another
- **Pharmacodynamic**: Additive or opposing effects
- **Pharmacokinetic**: Absorption/distribution/excretion effects

## Example Usage

**Input**: "Check interactions between aspirin, warfarin, and omeprazole"

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ddi_analysis", "query_type": "analyze", "drugs": "aspirin,warfarin,omeprazole"}'
```

**Expected result structure**:
```json
{
  "drugs_analyzed": ["aspirin", "warfarin", "omeprazole"],
  "total_pairs": 3,
  "interactions_found": 2,
  "severity_summary": {"Contraindicated": 1, "Precaution": 1, "Caution": 0},
  "interactions": [
    {
      "drug_a": "aspirin (D00109)",
      "drug_b": "warfarin (D00486)",
      "severity": "Contraindicated",
      "mechanism": "Increased bleeding risk..."
    },
    {
      "drug_a": "omeprazole (D00456)",
      "drug_b": "warfarin (D00486)",
      "severity": "Precaution",
      "mechanism": "CYP2C19 inhibition..."
    }
  ]
}
```

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Drug Not Found

**Symptom**: `find_drug` returns `kegg_id: null` or `analyze` shows drug in `unresolved` list.

**Solution**: Try alternative drug names or brand names. Check spelling. Some novel compounds may not be in KEGG database.

### No Interactions Found

**Symptom**: `interactions_found: 0`.

**Solution**: This may indicate no known DDIs between these drugs, or drugs not fully indexed in KEGG DDI. Consider clinical resources for comprehensive evaluation.

### Invalid Input

**Symptom**: Error message "At least 2 drugs required" or "Maximum 5 drugs allowed".

**Solution**: Provide between 2-5 drug names for analysis.

## Limitations

- KEGG DDI covers approved drugs only
- Novel compounds require predictive models (Way2Drug DDI-Pred)
- Severity classifications are database-defined
- Always consult clinical resources for patient-specific decisions
- Rate limiting applies (KEGG API: ~1 request/0.2 seconds)

## References

- KEGG DDI API: https://rest.kegg.jp/ddi/
- KEGG Drug Database: https://www.kegg.jp/kegg/drug/