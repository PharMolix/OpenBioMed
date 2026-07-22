---
name: iupac-name-identification-biot5
description: >
  Identify the IUPAC name of a molecule using BioT5 question answering model.
  Use this skill when:
  (1) User wants to find the IUPAC name of a molecule,
  (2) User asks "What is the IUPAC name?" or "What's the systematic name?",
  (3) User provides a SMILES string and wants the IUPAC nomenclature.
license: MIT
category: drug-discovery
tags: [iupac, molecule, nomenclature, question-answering, biot5]
---

# IUPAC Name Identification (BioT5)

This skill identifies the IUPAC name of a molecule by calling the OpenBioMed server API via curl.

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

- User asks for the IUPAC name of a molecule
- User provides a SMILES string and wants systematic nomenclature
- User asks "What is the IUPAC name?" or "What's the systematic name?"

## Workflow

### Step 1: Get the Molecule SMILES (if user provides a name)

Only needed when the user gives a molecule name (e.g., "aspirin") instead of a SMILES string. If the user already provides a SMILES, skip this step and use it directly in Step 2.

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

Extract the `molecule_preview` field — this is the SMILES string to use in Step 2.

### Step 2: Ask for IUPAC Name

Call the molecule question answering endpoint with the SMILES and the IUPAC question:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_question_answering", "model": "biot5", "molecule": "<SMILES>", "text": "What is the IUPAC name of this molecule?"}'
```

Response:
```json
{
  "task": "molecule_question_answering",
  "model": "biot5",
  "text": "<IUPAC name answer>"
}
```

Extract the `text` field — this contains the IUPAC name.

## Expected Outputs

| Input | API Endpoint | Response Field | Output |
|-------|-------------|---------------|--------|
| Molecule name | `/web_search/` | `molecule_preview` | SMILES string |
| SMILES + IUPAC question | `/run_pipeline/` | `text` | IUPAC name string |

## Example Usage

**Input**: "What is the IUPAC name of aspirin?"

**Step 1**: Get aspirin SMILES
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}'
```

Expected response:
```json
{"task": "molecule_name_request", "molecule": "...", "molecule_preview": "CC(=O)OC1=CC=CC=C1C(=O)O"}
```

**Step 2**: Ask for IUPAC name using the SMILES from Step 1
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_question_answering", "model": "biot5", "molecule": "CC(=O)OC1=CC=CC=C1C(=O)O", "text": "What is the IUPAC name of this molecule?"}'
```

Expected response:
```json
{"task": "molecule_question_answering", "model": "biot5", "text": "2-acetyloxybenzoic acid"}
```

**Final output**: "2-acetyloxybenzoic acid" (or similar systematic name)

## Model Options

The `molecule_question_answering` task supports multiple models via the `model` field:

| Model | Description |
|-------|-------------|
| `biot5` (default) | BioT5 model for biomedical QA |
| `molt5` | MolT5 model specialized for molecules |

To use `molt5`, change the `model` field in the curl command:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_question_answering", "model": "molt5", "molecule": "<SMILES>", "text": "What is the IUPAC name of this molecule?"}'
```

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Molecule Name Not Found

**Symptom**: `/web_search/` returns empty or null `molecule_preview`.

**Solution**: Ask user for the SMILES string directly and skip Step 1.

### QA Model Returns Empty Answer

**Symptom**: `/run_pipeline/` returns empty `text` field.

**Solution**:
- Try alternative question phrasing (e.g., "What is the systematic name?" or "Give the IUPAC nomenclature.")
- Switch model to `molt5`

## Notes

- IUPAC names generated by the model may not be the most standard form
- For complex molecules, the model may provide simplified names
- Cross-reference with PubChem or ChemDraw for verification