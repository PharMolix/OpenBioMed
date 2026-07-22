---
name: drug-lead-analysis
description: >
  Analyze drug candidate molecules for drug-likeness, ADMET properties, and safety profiles.
  Use this skill when:
  (1) Evaluating a molecule's potential as a drug candidate,
  (2) Checking drug-likeness scores (QED, Lipinski),
  (3) Predicting blood-brain barrier penetration,
  (4) Assessing side effects and ADMET properties,
  (5) Comparing multiple molecules for lead optimization.
license: MIT
category: drug-discovery
tags: [admet, drug-likeness, lead-optimization, molecular-properties]
---

# Drug Lead Analysis

Analyze drug candidate molecules for drug-likeness, ADMET properties, and safety profiles, via the OpenBioMed server API.

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

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/run_pipeline/` or `${OPENBIOMED_API_BASE_URL}/web_search/`.

## When to Use

- User asks to analyze a molecule for drug potential
- User provides a molecule name or SMILES and wants an evaluation
- User asks about drug-likeness, ADMET, BBB penetration, or side effects
- User wants to compare multiple molecules for lead optimization

## Workflow

### Step 1: Get the Molecule SMILES (if user provides a name)

Only needed when the user gives a molecule name instead of a SMILES string. If the user already provides a SMILES, skip this step.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "<molecule_name>"}'
```

Response:
```json
{
  "task": "molecule_name_request",
  "molecule": "<PubChem data>",
  "molecule_preview": "<SMILES string>"
}
```

Extract the `molecule_preview` field — this is the SMILES string for subsequent steps.

### Step 2: Run Drug Lead Analysis

Call the drug_lead_analysis endpoint with the SMILES. This returns QED, SA, LogP, and Lipinski scores in a single call:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "drug_lead_analysis", "molecule": "<SMILES>"}'
```

Response:
```json
{
  "task": "drug_lead_analysis",
  "model": null,
  "report": {
    "qed": 0.55,
    "sa": 1.58,
    "logp": 1.31,
    "lipinski": 4
  }
}
```

Extract the `report` field — this contains all four drug-likeness scores.

### Step 3: Predict ADMET Properties (Optional)

For ADMET predictions (BBBP penetration, SIDER side effects), call molecule_property_prediction separately:

```bash
# BBBP penetration prediction
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "<SMILES>", "dataset": "BBBP"}'

# SIDER side effect prediction
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "<SMILES>", "dataset": "SIDER"}'
```

Additional datasets: `caco2_wang` (Caco-2 permeability), `half_life_obach` (half-life), `ld50_zhu` (LD50 toxicity), `ClinTox` (clinical trial failure/FDA approval).

### Step 4: Summarize Findings

Combine the drug-likeness report from Step 2 and ADMET predictions from Step 3 into a structured assessment:

```
## Drug Lead Analysis Report: [Molecule Name]

### Drug-likeness Scores
| Metric | Value | Assessment |
|--------|-------|------------|
| QED | X.XX | [Good/Moderate/Poor] |
| SA Score | X.X | [Easy/Moderate/Hard to synthesize] |
| LogP | X.XX | [Optimal/High/Low] |
| Lipinski Violations | X | [Pass/Concern] |

### ADMET Properties
- Blood-Brain Barrier: [Penetrates/Does not penetrate]
- Predicted Side Effects: [List any predicted]

### Overall Assessment
[Summary of drug potential and recommendations]
```

## Expected Outputs

| Step | API Endpoint | Response Field | Output |
|------|-------------|---------------|--------|
| 1 (optional) | `/web_search/` | `molecule_preview` | SMILES string |
| 2 | `/run_pipeline/` | `report` | {qed, sa, logp, lipinski} |
| 3 (optional) | `/run_pipeline/` | `score` | BBBP/SIDER predictions |

## Interpretation Guide

### QED Score (Quantitative Estimate of Drug-likeness)
- **> 0.7**: Excellent drug-likeness
- **0.5 - 0.7**: Good drug-likeness
- **< 0.5**: Poor drug-likeness, may need optimization

### SA Score (Synthetic Accessibility)
- **1-3**: Easy to synthesize
- **3-6**: Moderate difficulty
- **6-10**: Difficult to synthesize

### LogP (Lipophilicity)
- **-0.4 to 5.6**: Optimal range for oral drugs
- **< -0.4**: Too hydrophilic, may have poor membrane permeability
- **> 5.6**: Too lipophilic, may have poor solubility

### Lipinski's Rule of Five
A "drug-like" molecule should have:
- Molecular weight ≤ 500 Da, LogP ≤ 5
- Hydrogen bond donors ≤ 5, acceptors ≤ 10

The `lipinski` value counts how many rules are satisfied (out of 4). 4 = ideal, 3 = acceptable, < 3 = concerning.

## Example Usage

**Input**: "Analyze aspirin as a drug candidate"

**Step 1**: Get aspirin SMILES
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}'
```

Expected response:
```json
{"task": "molecule_name_request", "molecule": "...", "molecule_preview": "CC(=O)Oc1ccccc1C(=O)O"}
```

**Step 2**: Drug lead analysis
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "drug_lead_analysis", "molecule": "CC(=O)Oc1ccccc1C(=O)O"}'
```

Expected response:
```json
{"task": "drug_lead_analysis", "model": null, "report": {"qed": 0.55, "sa": 1.58, "logp": 1.31, "lipinski": 4}}
```

**Step 3** (optional): ADMET predictions
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "BBBP"}'
```

Expected response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp", "score": "The blood-brain barrier penetration of the molecule is [0.188]"}
```

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Molecule Name Not Found

**Symptom**: `/web_search/` returns empty or null `molecule_preview`.

**Solution**: Ask user for the SMILES string directly and skip Step 1.

### drug_lead_analysis Not Available

**Symptom**: `/run_pipeline/` returns "drug_lead_analysis is currently not supported!"

**Solution**: The server may not have the `drug_lead_analysis` task registered. Fall back to calling `molecule_property_calculation` individually for QED, SA, LogP, and Lipinski (if supported), or compute via RDKit locally.

## Notes

- `drug_lead_analysis` computes drug-likeness metrics via RDKit (no ML model needed, fast response)
- ADMET predictions (BBBP, SIDER) require the `graphmvp` model and take longer
- For comparing multiple molecules, run Steps 2-3 for each molecule and summarize the comparison