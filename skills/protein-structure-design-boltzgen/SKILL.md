---
name: protein-structure-design-boltzgen
description: >
  Call interface for protein structure design via the /run_pipeline/ endpoint of any OpenBioMed-compatible HTTP service.
  Endpoint is configurable so this skill works against the OpenBioMed cloud service, a user-hosted instance, or a local dev server,
  independent of the underlying server implementation.
  Use this skill when:
  (1) Designing all-atom protein structures,
  (2) Creating protein binders for small molecules,
  (3) Generating novel protein sequences with specific structural constraints.
  Note: For structure prediction/validation, use protein_folding task.
license: MIT
category: protein-engineering
tags: [protein, structure-design, boltzgen, sequence-design, all-atom]
---

# Protein Structure Design - BoltzGen Call Interface

Call protein structure design via /run_pipeline/ interface using protein_folding task.

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

Privacy note: if the protein sequence or design data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## When to Use

- Designing novel protein sequences with specific structural properties
- Creating protein binders for small molecule targets
- Generating all-atom protein structures (not just backbone)
- Requiring precise binding geometries

**Note**: The `/run_pipeline/` interface uses `protein_folding` task for structure prediction. For direct structure prediction without design, use this same endpoint.

## API Parameters

**Required parameters:**
- `task`: "protein_folding"
- `protein`: Protein sequence in FASTA format

**Optional parameters:**
- `model`: Model name (default: esmfold)

```json
{
  "task": "protein_folding",
  "model": "esmfold_v1",
  "protein": "YOUR_AMINO_ACID_SEQUENCE"
}
```

## API Call Examples

### 1. Basic Protein Structure Prediction

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_folding",
  "model": "esmfold_v1",
  "protein": "YOUR_AMINO_ACID_SEQUENCE"
}'
```

### 2. Predict Structure of Enzyme

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_folding",
  "protein": "YOUR_ENZYME_SEQUENCE"
}'
```

### 3. Structure Validation for Designed Protein

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_folding",
  "model": "esmfold_v1",
  "protein": "YOUR_DESIGNED_SEQUENCE"
}'
```

## Limitations

- Max sequence length: ~1500 amino acids (depends on model)
- Longer sequences may be truncated
- Requires GPU for reasonable inference time
- For complex design tasks (ligand binding, side-chain optimization), use BoltzGen CLI directly

## Related Skills

- `structure-prediction-boltz-2`: For advanced structure prediction with Boltz-2
- `protein-function-prediction`: For function prediction of designed proteins
- `protein-binding-site-prediction`: For identifying binding sites in predicted structures
