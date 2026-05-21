---
name: uniprot-query
description: >
  Query UniProt database for protein sequences, metadata, and search by criteria.
  Use this skill when:
  (1) Looking up protein information by UniProt accession ID,
  (2) Searching proteins by gene name, organism, function, or disease,
  (3) Retrieving comprehensive protein metadata including domains, PTMs, and annotations.
license: MIT
category: knowledge-retrieval
tags: [uniprot, protein, database, metadata, sequence]
---

# UniProt Query

Query the UniProt knowledgebase for comprehensive protein information, via the OpenBioMed server API.

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

Privacy note: if the protein sequence is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/web_search/`.

## When to Use

- Look up protein by UniProt accession (e.g., P00533 for EGFR)
- Search proteins by gene name, organism, or keywords
- Retrieve protein sequences and structural annotations
- Get protein metadata: function, domains, diseases

## Workflow

### Step 1: Look Up Protein by UniProt Accession ID

Call the protein_uniprot_request endpoint with the accession ID:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "protein_uniprot_request", "query": "<UniProt_accession>"}'
```

Response:
```json
{
  "task": "protein_uniprot_request",
  "protein": "<protein data or file path>",
  "protein_preview": "<sequence preview or FASTA>"
}
```

Extract the `protein` and `protein_preview` fields — these contain the protein data.

### Step 2: Fetch Full Metadata from UniProt REST API (Optional)

For comprehensive metadata beyond the basic sequence, call the UniProt REST API directly:

```bash
curl -s "https://rest.uniprot.org/uniprotkb/<UniProt_accession>?format=json"
```

This returns detailed metadata including function, domains, diseases, PTMs, and annotations.

### Step 3: Search by Criteria (Optional)

Search UniProt by gene name, organism, keywords, or disease. This is a direct call to the UniProt search API:

```bash
curl -s "https://rest.uniprot.org/uniprotkb/search?query=gene_exact:EGFR+AND+organism_id:9606+AND+reviewed:true&fields=accession,gene_primary,protein_name,organism_name,length&format=json&size=10"
```

## Expected Outputs

| Step | API Endpoint | Response Field | Output |
|------|-------------|---------------|--------|
| 1 | `/web_search/` | `protein`, `protein_preview` | Protein sequence and basic data |
| 2 (optional) | UniProt REST API | Full JSON | Comprehensive metadata (function, domains, diseases) |
| 3 (optional) | UniProt REST API | Search results JSON | List of matching proteins |

## Query Syntax Reference (for Step 3)

| Field | Example | Description |
|-------|---------|-------------|
| `gene_exact` | `gene_exact:EGFR` | Exact gene name match |
| `gene` | `gene:BRCA` | Gene name (partial match) |
| `organism_id` | `organism_id:9606` | Organism by TaxID |
| `organism` | `organism:"Homo sapiens"` | Organism by name |
| `protein_name` | `protein_name:kinase` | Protein name search |
| `keyword` | `keyword:Kinase` | UniProt keyword |
| `cc_disease` | `cc_disease:diabetes` | Disease association |
| `reviewed` | `reviewed:true` | Swiss-Prot only (curated) |

**Common Organism IDs**: Human (9606), Mouse (10090), SARS-CoV-2 (2697049), E. coli (83333)

**Combine queries**: Use `AND`, `OR` — e.g., `gene_exact:EGFR AND organism_id:9606 AND reviewed:true`

## Example Usage

**Input**: "Tell me about the EGFR protein (P00533)"

**Step 1**: Look up protein by accession
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "protein_uniprot_request", "query": "P00533"}'
```

Expected response:
```json
{"task": "protein_uniprot_request", "protein": "...", "protein_preview": "..."}
```

**Step 2** (optional): Fetch full metadata
```bash
curl -s "https://rest.uniprot.org/uniprotkb/P00533?format=json"
```

This returns the complete UniProt entry with function, domains, diseases, etc.

**Input**: "Search for human kinases"

**Step 3**: Search by criteria
```bash
curl -s "https://rest.uniprot.org/uniprotkb/search?query=keyword:Kinase+AND+organism_id:9606+AND+reviewed:true&fields=accession,gene_primary,protein_name,organism_name,length&format=json&size=10"
```

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout for OpenBioMed server.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Accession Not Found

**Symptom**: `/web_search/` returns empty or error response for the accession ID.

**Solution**: Verify the UniProt ID format (e.g., `P00533`, not `EGFR`). Try searching by gene name using Step 3 instead.

### UniProt REST API Errors

**Symptom**: UniProt API returns 404 or no results.

**Solution**: Broaden query, remove `reviewed:true`, check organism ID. UniProt rate limit is ~10 requests/second — retry after a short wait.

## Notes

- Step 1 (OpenBioMed API) provides protein sequence and basic data
- Steps 2-3 (UniProt REST API) provide comprehensive metadata and search — these are external calls not dependent on the OpenBioMed server
- For proprietary protein sequences, always use a self-hosted OpenBioMed endpoint