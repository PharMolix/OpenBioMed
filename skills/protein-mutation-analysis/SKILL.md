---
name: protein-mutation-analysis
description: >
  Call interface for protein mutation analysis via the /run_pipeline/ endpoint of any OpenBioMed-compatible HTTP service.
  Supports two tasks: mutation_explanation and mutation_engineering. Endpoint is configurable so this skill works against
  the OpenBioMed cloud service, a user-hosted instance, or a local dev server, independent of the underlying server implementation.
  Use this skill when:
  (1) Analyzing mutation effects on protein function,
  (2) Understanding how specific mutations affect protein structure and activity,
  (3) Explaining disease-associated genetic variants.
license: MIT
category: protein-engineering
tags: [protein, mutation, variant-analysis, mutaplm, genetic-variants]
---

# Protein Mutation Analysis Call Interface

Call protein mutation analysis via /run_pipeline/ interface. Supports mutation explanation and mutation engineering.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520` or `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: if the protein sequence or mutation data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## When to Use

- User provides a UniProt ID and mutation (e.g., "P04637 R248Q")
- User wants to understand the effect of a specific mutation
- User needs to explain disease-associated genetic variants

## Two Available Tasks

### Task 1: Mutation Explanation (mutation_explanation)

Explain the functional impact of a specific mutation.

**Required parameters:**
- `task`: "mutation_explanation"
- `protein`: Protein sequence in FASTA format or UniProt accession
- `mutation`: Mutation in format "OriginalAA + Position + MutantAA" (e.g., "R248Q")

**Optional parameters:**
- `model`: Model name (default: mutaplm)

```json
{
  "task": "mutation_explanation",
  "model": "mutaplm",
  "protein": "YOUR_PROTEIN_SEQUENCE",
  "mutation": "R248Q"
}
```

### Task 2: Mutation Engineering (mutation_engineering)

Generate protein mutations based on a text description (e.g., "improve stability", "increase affinity").

**Required parameters:**
- `task`: "mutation_engineering"
- `protein`: Protein sequence in FASTA format or UniProt accession
- `text`: Description of desired mutation properties

**Optional parameters:**
- `model`: Model name (default: mutaplm)

```json
{
  "task": "mutation_engineering",
  "model": "mutaplm",
  "protein": "YOUR_PROTEIN_SEQUENCE",
  "text": "improve thermal stability"
}
```

**Note**: The response includes a `protein` field containing a file path on the server. External agents cannot access this path directly. Use the `read_protein_file` task below to get the actual protein content.

### Task 3: Read Protein File Content (read_protein_file)

After receiving a protein file path from mutation_engineering, call this task to get the actual sequence content accessible to external agents.

**Required parameters:**
- `task`: "read_protein_file"
- `protein`: Protein file path from mutation_engineering response
- `value`: "true" to include PDB content (3D structure), "false" for sequence only

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "<protein_file_path>", "value": "true"}'
```

Response:
```json
{
  "task": "read_protein_file",
  "sequence": "<protein sequence>",
  "name": "<protein name>",
  "pdb_content": "<PDB file content for 3D structure>",
  "structure_note": "<note if protein has no 3D structure>"
}
```

## API Call Examples

### 1. Mutation Explanation - EGFR T790M

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "mutation_explanation",
  "model": "mutaplm",
  "protein": "MRGPRGSRCQWLRRGNSSKGRQV",
  "mutation": "T790M"
}'
```

### 2. Mutation Engineering - Improve Stability

**Step 1**: Generate mutations

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "mutation_engineering",
  "model": "mutaplm",
  "protein": "YOUR_PROTEIN_SEQUENCE",
  "text": "improve thermal stability"
}'
```

Expected response:
```json
{
  "task": "mutation_engineering",
  "model": "mutaplm",
  "mutation": ["M1A", "M2B", ...],
  "protein": "./tmp/protein_xxx.pkl",
  "protein_preview": "MKT..."
}
```

**Step 2**: Read protein file content (for external agents)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "./tmp/protein_xxx.pkl", "value": "true"}'
```

Expected response:
```json
{
  "task": "read_protein_file",
  "sequence": "MKT...",
  "name": "engineered_protein",
  "pdb_content": "..."
}
```

## Common Use Cases

### 1. Cancer Mutation Analysis - EGFR L858R

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "mutation_explanation",
  "model": "mutaplm",
  "protein": "YOUR_EGFR_SEQUENCE",
  "mutation": "L858R"
}'
```

### 2. Drug Resistance Mutation

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "mutation_explanation",
  "model": "mutaplm",
  "protein": "YOUR_PROTEIN_SEQUENCE",
  "mutation": "G12D"
}'
```

### 3. Generate Stabilizing Mutations

**Step 1**: Generate mutations
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "mutation_engineering",
  "model": "mutaplm",
  "protein": "YOUR_PROTEIN_SEQUENCE",
  "text": "increase protein stability"
}'
```

**Step 2**: Read protein file content (for external agents)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "<protein_file_path>", "value": "true"}'
```

### 4. Generate Affinity-Enhancing Mutations

**Step 1**: Generate mutations
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "mutation_engineering",
  "model": "mutaplm",
  "protein": "YOUR_PROTEIN_SEQUENCE",
  "text": "enhance binding affinity to target"
}'
```

**Step 2**: Read protein file content (for external agents)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "<protein_file_path>", "value": "true"}'
```

## Limitations

- Mutation format must be: OriginalAA + Position + MutantAA (e.g., "R248Q")
- Longer protein sequences may take more time to process
- Novel mutations in unstructured regions may have lower prediction accuracy

## Related Skills

- `protein-function-prediction`: For predicting protein function
- `protein-structure-design-boltzgen`: For 3D structure prediction
- `uniprot-query`: For retrieving protein sequences from UniProt
