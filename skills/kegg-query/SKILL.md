---
name: kegg-query
description: >
  Query KEGG database for drug information, pathway analysis, and disease-drug-target discovery.
  Use this skill when:
  (1) Looking up drug information including efficacy, targets, metabolism, and interactions,
  (2) Analyzing metabolic or signaling pathways to retrieve genes, compounds, and modules,
  (3) Discovering disease-associated drugs, genes, and pathways for drug repurposing.
license: MIT
category: knowledge-retrieval
tags: [kegg, pathway, drug, disease, target, bioinformatics]
---

# KEGG Query

Query KEGG database for drug, pathway, and disease information via the OpenBioMed server API.

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

Privacy note: KEGG is a public database, so there are no privacy concerns with using either endpoint.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). KEGG queries use the endpoint `${OPENBIOMED_API_BASE_URL}/run_pipeline/` with `task: "kegg_query"`.

## When to Use

- **Drug Lookup**: Retrieve drug efficacy, targets, metabolism, drug-drug interactions
- **Pathway Analysis**: Get pathway genes, compounds, modules, and related pathways
- **Disease Discovery**: Find disease-associated drugs, genes, and therapeutic targets

## Workflow

### Use Case 1: Drug Information Lookup

Search for a drug by name, then retrieve full entry details.

**Step 1: Find drug by name** (`query_type: "find"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "find", "database": "drug", "query": "aspirin"}'
```

Response:
```json
{
  "task": "kegg_query",
  "query_type": "find",
  "results": [
    {"entry_id": "dr:D00109", "description": "Aspirin (JP18/USP); Acetylsalicylic acid; ..."}
  ]
}
```

**Step 2: Get full drug entry** (`query_type: "get"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "get", "entry_id": "D00109"}'
```

The entry_id is auto-formatted (e.g., `D00109` → `dr:D00109`). You can also use the full format directly:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "get", "entry_id": "dr:D00109"}'
```

Response:
```json
{
  "task": "kegg_query",
  "query_type": "get",
  "results": [{"entry_id": "dr:D00109", "raw_text": "...full KEGG entry text..."}]
}
```

The `raw_text` field contains the complete KEGG DRUG entry. Parse key fields like NAME, FORMULA, EFFICACY, TARGET, PATHWAY, DISEASE from this text (see Expected Outputs below).

**Optional: Get molecular structure** (`option: "mol"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "get", "entry_id": "D00109", "option": "mol"}'
```

### Use Case 2: Pathway Analysis

Retrieve pathway entry with genes, compounds, and modules.

**Get pathway by ID** (`query_type: "get"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "get", "entry_id": "hsa00010"}'
```

Response:
```json
{
  "task": "kegg_query",
  "query_type": "get",
  "results": [{"entry_id": "hsa00010", "raw_text": "...full pathway entry with genes, compounds, modules..."}]
}
```

**Find pathway by keyword** (`query_type: "find"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "find", "database": "pathway", "query": "glycolysis"}'
```

**Cross-reference pathway with compounds** (`query_type: "link"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "link", "target_db": "compound", "source_id": "hsa00010"}'
```

Response:
```json
{
  "task": "kegg_query",
  "query_type": "link",
  "results": [
    {"source_id": "hsa00010", "target_id": "cpd:C00031"},
    {"source_id": "hsa00010", "target_id": "cpd:C00022"}
  ]
}
```

### Use Case 3: Disease-Drug-Target Discovery

Find diseases by keyword, then retrieve associated drugs and targets.

**Step 1: Search for disease** (`query_type: "find"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "find", "database": "disease", "query": "diabetes"}'
```

**Step 2: Get disease details** (`query_type: "get"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "get", "entry_id": "H00409"}'
```

The entry_id is auto-formatted (`H00409` → `ds:H00409`). The `raw_text` contains the full KEGG DISEASE entry with associated genes, drugs, and pathways.

**Step 3: Cross-reference disease with drugs** (`query_type: "link"`):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "kegg_query", "query_type": "link", "target_db": "drug", "source_id": "ds:H00409"}'
```

## Expected Outputs

### Find Operation

Returns a list of matching entries:

| Field | Description |
|-------|-------------|
| `entry_id` | KEGG entry identifier (e.g., `dr:D00109`) |
| `description` | Entry name and synonyms |

### Get Operation

Returns a single entry with `raw_text` containing the full KEGG flat-file format. Key sections to parse:

**Drug Entry** fields: ENTRY, NAME, FORMULA, EXACT_MASS, MOL_WEIGHT, EFFICACY, TARGET, PATHWAY, DISEASE, DBLINKS

**Pathway Entry** fields: ENTRY, NAME, DESCRIPTION, CLASS, ORGANISM, MODULE, GENE, COMPOUND, REL_PATHWAY

**Disease Entry** fields: ENTRY, NAME, DESCRIPTION, CATEGORY, GENE, DRUG, PATHWAY, NETWORK, DBLINKS

### Link Operation

Returns cross-reference pairs:

| Field | Description |
|-------|-------------|
| `source_id` | Source entry ID |
| `target_id` | Target entry ID |

## Parameters Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | str (required) | Must be `"kegg_query"` |
| `query_type` | str (required) | `"find"`, `"get"`, or `"link"` |
| `database` | str | KEGG database for find: `"drug"`, `"compound"`, `"disease"`, `"pathway"`, `"genes"`, `"enzyme"`, `"ko"` (default: `"drug"`) |
| `query` | str | Search keyword for find; also fallback for entry_id/source_id |
| `entry_id` | str | Entry ID for get (auto-formatted: `D00109` → `dr:D00109`, `C00031` → `cpd:C00031`, `H00409` → `ds:H00409`) |
| `option` | str | Optional format for get: `"aaseq"`, `"ntseq"`, `"mol"`, `"image"`, `"kgml"` |
| `target_db` | str | Target database for link (e.g., `"drug"`, `"compound"`, `"pathway"`) (default: `"drug"`) |
| `source_id` | str | Source entry/database ID for link; fallback to `query` if not provided |

## Entry ID Formats

Auto-formatting rules (applied automatically by the tool):

| Raw ID | Auto-formatted | Database |
|--------|---------------|----------|
| `D00109` | `dr:D00109` | Drug |
| `C00031` | `cpd:C00031` | Compound |
| `H00409` | `ds:H00409` | Disease |
| `hsa00010` | `hsa00010` (no change) | Pathway (organism-specific) |
| `map00010` | `map00010` (no change) | Pathway (reference) |

You can also provide the full prefixed format directly (`dr:D00109`, `cpd:C00031`, etc.).

## Error Handling

### Entry Not Found

**Symptom**: `get` returns empty or very short `raw_text`.

**Solution**: Use `find` first to get the exact entry ID, then use `get` with that ID.

### No Search Results

**Symptom**: `find` returns empty results list.

**Solution**: Try alternative keywords or different database names. Use simpler terms (e.g., "cancer" instead of "carcinoma").

### Rate Limiting

**Symptom**: Repeated requests fail or timeout.

**Solution**: KEGG allows ~5 requests/second. The server implements rate limiting (5 calls/second). Add delays between rapid queries.

### Timeout

**Symptom**: curl returns timeout after long wait.

**Solution**: KEGG entries can be large (pathway entries have hundreds of genes). Reduce query complexity.

### Invalid ID Format

**Symptom**: `get` returns error for malformed entry_id.

**Solution**: Use auto-formatting (e.g., provide `D00109` instead of `dr:D00109`), or use the full prefixed format.

## Notes

- KEGG queries use the `/run_pipeline/` endpoint with `task: "kegg_query"`
- The `get` operation returns raw KEGG flat-file text in `raw_text` — parse relevant sections as needed
- Auto-formatting of entry IDs is handled tool-side, so `D00109` and `dr:D00109` both work
- The `link` operation is useful for discovering cross-references (e.g., disease → drugs, pathway → compounds)
- KEGG API rate limit: 5 requests/second (enforced server-side)
