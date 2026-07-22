---
name: admet-prediction
description: >
  Predict comprehensive ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
  properties for drug candidate molecules using GraphMVP ensemble models.
  Use this skill when:
  (1) Predicting blood-brain barrier penetration,
  (2) Assessing side effect profiles,
  (3) Estimating Caco-2 permeability, half-life, or LD50 toxicity,
  (4) Evaluating drug-likeness and safety of molecules.
license: MIT
category: admet-prediction
tags: [admet, toxicity, drug-discovery, pharmacokinetics, graphmvp]
---

# ADMET Prediction

Predict comprehensive ADMET properties for drug candidate molecules via the OpenBioMed server API.

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

- User asks to predict ADMET properties for a molecule
- User provides a drug candidate and wants safety assessment
- User needs blood-brain barrier penetration prediction
- User wants to evaluate toxicity (LD50) or side effects (SIDER)
- User requests pharmacokinetic properties (half-life, Caco-2)

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

### Step 2: Run Classification Predictions (BBBP + SIDER)

BBBP and SIDER use the `graphmvp` classification model:

```bash
# Blood-brain barrier penetration
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "<SMILES>", "dataset": "BBBP"}'
```

Response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp", "score": "The blood-brain barrier penetration of the molecule is [0.188]"}
```

```bash
# Side effects (27 categories)
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "<SMILES>", "dataset": "SIDER"}'
```

Response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp", "score": "<27 side effect probabilities>"}
```

### Step 3: Run Regression Predictions (Caco-2 + Half-life + LD50)

Caco-2, half-life, and LD50 use the `graphmvp_regression` model:

```bash
# Caco-2 permeability
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp_regression", "molecule": "<SMILES>", "dataset": "caco2_wang"}'
```

Response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp_regression", "score": "The rate of drug passing through the Caco-2 cells is [-4.678]"}
```

```bash
# Half-life
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp_regression", "molecule": "<SMILES>", "dataset": "half_life_obach"}'
```

Response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp_regression", "score": "The half-life of the molecule is [-7.0594]"}
```

```bash
# LD50 toxicity
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp_regression", "molecule": "<SMILES>", "dataset": "ld50_zhu"}'
```

Response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp_regression", "score": "The most conservative dose of the molecule that can lead to lethal adverse effects is [2.0629]"}
```

### Step 4: Summarize Findings

Combine all predictions into a structured ADMET report.

## Expected Outputs

| Dataset | Model | Response Field | Output |
|---------|-------|---------------|--------|
| BBBP | graphmvp | score | BBB penetration probability |
| SIDER | graphmvp | score | 27 side effect probabilities |
| caco2_wang | graphmvp_regression | score | Log permeability (cm/s) |
| half_life_obach | graphmvp_regression | score | Log half-life (hours) |
| ld50_zhu | graphmvp_regression | score | Log LD50 (mg/kg) |

## Interpretation Guide

### BBB Penetration

| Value | Interpretation |
|-------|----------------|
| > 0.5 | Likely crosses BBB |
| < 0.5 | Unlikely to cross BBB |

### Caco-2 Permeability

| Value (log cm/s) | Interpretation |
|------------------|----------------|
| > -5 | High absorption |
| -6 to -5 | Moderate absorption |
| < -6 | Low absorption |

### LD50 Toxicity

| Value (log mg/kg) | Toxicity Level |
|-------------------|----------------|
| < 1 | Highly toxic (<10 mg/kg) |
| 1-2 | Moderately toxic (10-100 mg/kg) |
| 2-3 | Slightly toxic (100-1000 mg/kg) |
| > 3 | Low toxicity (>1000 mg/kg) |

### SIDER Side Effects

Values range 0-1. Categories with **> 0.7** indicate high risk of that side effect.

## Example Usage

**Input**: "Predict ADMET properties for aspirin"

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

**Step 2**: Classification predictions (BBBP + SIDER)
```bash
# BBBP
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "BBBP"}'

# SIDER
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "SIDER"}'
```

**Step 3**: Regression predictions (Caco-2 + Half-life + LD50)
```bash
# Caco-2
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp_regression", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "caco2_wang"}'

# Half-life
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp_regression", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "half_life_obach"}'

# LD50
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp_regression", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "ld50_zhu"}'
```

**Expected results** (aspirin):
- BBB Penetration: 0.19 (does NOT cross BBB)
- Caco-2: -4.68 (moderate absorption)
- Half-life: -7.06 (short half-life)
- LD50: 2.06 (moderate toxicity ~115 mg/kg)
- Top Side Effects: Skin disorders (0.80), Nervous system (0.78), Gastrointestinal (0.78)

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Molecule Name Not Found

**Symptom**: `/web_search/` returns empty or null `molecule_preview`.

**Solution**: Ask user for the SMILES string directly and skip Step 1.

### Dataset Not Supported

**Symptom**: `/run_pipeline/` returns error for a dataset name.

**Solution**: The server may not have the checkpoint for that dataset. Check which datasets are available and only request supported ones. Classification datasets: BBBP, SIDER, ClinTox. Regression datasets: caco2_wang, half_life_obach, ld50_zhu.

### Invalid SMILES

**Symptom**: Prediction fails for a malformed SMILES string.

**Solution**: Validate SMILES format. Use molecule name lookup via Step 1 instead.

## Notes

- Classification tasks (BBBP, SIDER) use model `graphmvp`; regression tasks (caco-2, half-life, LD50) use model `graphmvp_regression`
- Each prediction is an independent API call — the skill makes 5 calls total for full ADMET profile
- SIDER returns 27 side effect category probabilities — flag categories > 0.7 as high risk