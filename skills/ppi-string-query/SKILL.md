---
name: ppi-string-query
description: >
  Query STRING database for protein-protein interactions with confidence scores.
  Use this skill when:
  (1) Finding interaction partners for a protein of interest,
  (2) Retrieving confidence scores for protein-protein interactions,
  (3) Building protein interaction networks for pathway analysis.
license: MIT
category: knowledge-retrieval
tags: [string, protein-protein-interaction, ppi, network, interactions]
---

# STRING Protein-Protein Interaction Query

Query the STRING database for protein-protein interactions with confidence scores via the OpenBioMed server API.

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

Privacy note: STRING is a public database, so there are no privacy concerns with using either endpoint.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). STRING PPI queries use the endpoint `${OPENBIOMED_API_BASE_URL}/run_pipeline/` with `task: "ppi_string_request"`.

## When to Use

- Find interaction partners for a protein (by UniProt ID)
- Retrieve confidence scores for PPIs (experimental, text mining, database)
- Build protein interaction networks for pathway analysis
- Identify potential protein complexes or functional modules

## Workflow

### Basic Query: Find Interaction Partners

Query STRING for interaction partners of a protein using its UniProt ID:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ppi_string_request", "uniprot_id": "P04637"}'
```

Response:
```json
{
  "task": "ppi_string_request",
  "uniprot_id": "P04637",
  "results": [
    {
      "query_protein": "TP53",
      "partner_string_id": "9606.ENSP00000340989",
      "partner_gene": "SFN",
      "combined_score": 0.999,
      "scores": {
        "experimental": 0.981,
        "text_mining": 0.859,
        "database": 0.75,
        "coexpression": 0.0,
        "phylogenetic": 0.0,
        "gene_fusion": 0.0,
        "neighborhood": 0.0
      },
      "ncbi_taxon_id": 9606
    }
  ]
}
```

### Custom Parameters

Adjust confidence threshold, species, and result limit:

**High confidence interactions only (score >= 700, limit 20)**:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ppi_string_request", "uniprot_id": "P04637", "species": 9606, "required_score": 700, "limit": 20}'
```

**Highest confidence only (score >= 900)**:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ppi_string_request", "uniprot_id": "P24941", "species": 9606, "required_score": 900, "limit": 10}'
```

**Mouse protein interactions (species 10090)**:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ppi_string_request", "uniprot_id": "Q8BIE6", "species": 10090, "required_score": 700}'
```

### Multi-step Workflow: PPI Network Analysis

**Step 1**: Query STRING for primary interactors:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ppi_string_request", "uniprot_id": "P04637", "required_score": 700, "limit": 15}'
```

**Step 2**: For each high-confidence partner, query STRING again to expand the network (repeat for top interactors):

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "ppi_string_request", "uniprot_id": "P00533", "required_score": 700, "limit": 15}'
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str (required) | Must be `"ppi_string_request"` |
| `uniprot_id` | str (required) | UniProt accession (e.g., `P04637` for TP53) |
| `species` | int | 9606 | NCBI taxonomy ID (9606=human, 10090=mouse, etc.) |
| `required_score` | int | 700 | Min confidence threshold (150/400/700/900) |
| `limit` | int | 50 | Max interactors to return |

## Confidence Score Thresholds

| Score | Level | Use Case |
|-------|-------|----------|
| 150 | Low | Exploratory analysis, broad network |
| 400 | Medium | Balanced retrieval |
| 700 | High | Reliable interactions (default) |
| 900 | Highest | Very confident only, core interactors |

## Expected Output

Each result entry contains:

| Field | Description |
|-------|-------------|
| `query_protein` | Gene symbol of the query protein |
| `partner_string_id` | STRING identifier of the partner |
| `partner_gene` | Gene symbol of the interaction partner |
| `combined_score` | Weighted confidence score (0-1) |
| `scores.experimental` | Evidence from wet-lab experiments |
| `scores.text_mining` | Literature co-occurrence evidence |
| `scores.database` | Curated database evidence (BioGRID, etc.) |
| `scores.coexpression` | Expression correlation evidence |
| `scores.phylogenetic` | Phylogenetic profile evidence |
| `scores.gene_fusion` | Gene fusion evidence |
| `scores.neighborhood` | Genomic proximity evidence |
| `ncbi_taxon_id` | NCBI taxonomy ID |

## Score Interpretation

| Score Type | Source | Range |
|------------|--------|-------|
| `combined_score` | Probabilistic integration: `1 - Product(1 - channel)` | 0-1 |
| `experimental` | Wet-lab experiments (BioGRID, IntAct) | 0-1 |
| `text_mining` | Literature co-occurrence | 0-1 |
| `database` | Curated databases (BioGRID, DIP, MINT) | 0-1 |
| `coexpression` | Expression correlation across conditions | 0-1 |
| `phylogenetic` | Phylogenetic profile similarity | 0-1 |
| `gene_fusion` | Fusion events across genomes | 0-1 |
| `neighborhood` | Genomic proximity in prokaryotes | 0-1 |

## Common Organism IDs

| Organism | Taxonomy ID |
|----------|-------------|
| Human | 9606 |
| Mouse | 10090 |
| Rat | 10116 |
| E. coli | 83333 |
| S. cerevisiae | 4932 |

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### No Interactions Found

**Symptom**: Empty results list.

**Solution**: Lower `required_score` threshold (try 400 or 150). Some proteins, especially less-studied ones, may only have low-confidence interactions.

### UniProt ID Not Recognized

**Symptom**: Error response from STRING.

**Solution**: Verify the UniProt ID format (e.g., `P04637`, not `TP53`). STRING uses UniProt accessions as identifiers, not gene symbols. Check the correct ID at https://www.uniprot.org/.

### Rate Limiting

**Symptom**: Repeated requests fail or timeout.

**Solution**: STRING allows ~5 requests/second. The server implements rate limiting. Add delays between rapid queries.

### Wrong Species

**Symptom**: Unexpected or empty results for a known protein.

**Solution**: Check the NCBI taxonomy ID. A human protein queried with `species=10090` (mouse) may not return results. Use the correct species ID for the organism.

## Notes

- STRING PPI queries use the `/run_pipeline/` endpoint with `task: "ppi_string_request"`
- Default parameters: `species=9606` (human), `required_score=700` (high confidence), `limit=50`
- The combined score is a probabilistic integration of all seven evidence channels
- For network analysis, start with high confidence (700) and expand by querying top partners iteratively