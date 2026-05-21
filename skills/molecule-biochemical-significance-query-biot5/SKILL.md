---
name: molecule-biochemical-significance-query-biot5
description: >
  Query a molecule's biochemical significance and roles in biology and chemistry
  using BioT5 multi-modal model.
  Use this skill when:
  (1) Understanding a molecule's biological roles and functions,
  (2) Describing a molecule's chemical significance and applications,
  (3) Getting natural language explanations of molecular properties,
  (4) Summarizing what a molecule is used for or its metabolic relevance.
license: MIT
category: multi-modal-reasoning
tags: [molecule, question-answering, biochemistry, biot5, multi-modal]
---

# Molecule Biochemical Significance Query

Query a molecule's biochemical significance by calling the OpenBioMed server API via curl.

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

Privacy note: if the molecule data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/run_pipeline/` or `${OPENBIOMED_API_BASE_URL}/web_search/`.

## When to Use

- User asks about a molecule's biological roles or functions
- User wants to understand what a molecule is used for
- User requests natural language description of molecular properties
- User asks about a molecule's metabolic or chemical significance

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

### Step 2: Ask About Biochemical Significance

Call the molecule question answering endpoint with the SMILES and a question about biochemical significance:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_question_answering", "model": "biot5", "molecule": "<SMILES>", "text": "I am interested in understanding the molecule biochemical significance; can you describe its roles in biology and chemistry?"}'
```

Response:
```json
{
  "task": "molecule_question_answering",
  "model": "biot5",
  "text": "<natural language answer>"
}
```

Extract the `text` field — this contains the biochemical significance description.

## Expected Outputs

| Input | API Endpoint | Response Field | Output |
|-------|-------------|---------------|--------|
| Molecule name | `/web_search/` | `molecule_preview` | SMILES string |
| SMILES + significance question | `/run_pipeline/` | `text` | Biochemical significance description |

### Example Outputs

| Molecule | SMILES | Output |
|----------|--------|--------|
| Heptylfuran | `CCCCCCCc1ccco1` | "flavouring agent; fragrance; metabolite" |
| Aspirin | `CC(=O)OC1=CC=CC=C1C(=O)O` | "analgesic; anti-inflammatory; antipyretic" |

## Example Usage

**Input**: "What is the biochemical significance of heptylfuran?"

**Step 1** (skipped — SMILES provided directly): Use SMILES `CCCCCCCc1ccco1`

**Step 2**: Ask about biochemical significance
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_question_answering", "model": "biot5", "molecule": "CCCCCCCc1ccco1", "text": "I am interested in understanding the molecule biochemical significance; can you describe its roles in biology and chemistry?"}'
```

Expected response:
```json
{"task": "molecule_question_answering", "model": "biot5", "text": "flavouring agent; fragrance; metabolite"}
```

**Input**: "What is aspirin used for?"

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

**Step 2**: Ask about biochemical significance
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_question_answering", "model": "biot5", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "text": "I am interested in understanding the molecule biochemical significance; can you describe its roles in biology and chemistry?"}'
```

Expected response:
```json
{"task": "molecule_question_answering", "model": "biot5", "text": "analgesic; anti-inflammatory; antipyretic"}
```

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
  -d '{"task": "molecule_question_answering", "model": "molt5", "molecule": "<SMILES>", "text": "I am interested in understanding the molecule biochemical significance; can you describe its roles in biology and chemistry?"}'
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
- Try alternative question phrasing (e.g., "Describe the biological roles of this molecule." or "What is this molecule used for?")
- Switch model to `molt5`

## Notes

- The model's answers are concise summaries, not detailed explanations
- For complex molecules, the answer may focus on the most prominent roles
- Cross-reference with PubChem or ChEMBL for comprehensive biochemical profiles