---
name: biomedical-literature-search
description: >
  Search biomedical literature from PubMed and bioRxiv for research papers.
  Use this skill when:
  (1) Finding research papers on a specific topic or disease,
  (2) Retrieving recent preprints from bioRxiv,
  (3) Getting paper titles, abstracts, and metadata,
  (4) Literature review for drug discovery or biomedical research.
license: MIT
---

# Biomedical Literature Search

Search PubMed and bioRxiv for biomedical research papers with titles and abstracts via the OpenBioMed server API.

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

Privacy note: if the search query data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## When to Use

- Find research papers on a specific biomedical topic
- Retrieve recent preprints from bioRxiv
- Get paper titles, abstracts, authors, and links
- Literature review for drug discovery or biomedical research

## API Query Types

The `literature_search` task supports the following query types via `query_type` parameter:

| query_type | Description | Required Params | Optional Params |
|------------|-------------|-----------------|-----------------|
| `pubmed_search` | Search PubMed by keywords | `query` | `max_results` |
| `pubmed_fetch` | Fetch paper details by PMIDs | `pmids` | - |
| `biorxiv_fetch` | Fetch bioRxiv papers by date range | - | `start_date`, `end_date`, `days` |
| `biorxiv_category` | Fetch bioRxiv papers by category | - | `category`, `days` |

Note: `pmids` can be a comma-separated string or a list. bioRxiv does not support keyword search - use date range or category filters.

## Workflow

### Step 1: PubMed Search (Keyword-based)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "pubmed_search", "query": "PD-1 inhibitor cancer", "max_results": 10}'
```

Response:
```json
{
  "task": "literature_search",
  "query_type": "pubmed_search",
  "results": [{
    "query": "PD-1 inhibitor cancer",
    "papers_found": 10,
    "pmids": ["38412345", "38412346", ...],
    "papers": [
      {
        "title": "PD-1 blockade in cancer immunotherapy",
        "authors": "Smith J, Doe A",
        "abstract": "Full abstract text...",
        "doi": "10.1234/journal.2024",
        "pmid": "38412345",
        "date": "2024-Jan",
        "journal": "Nature Medicine",
        "link": "https://pubmed.ncbi.nlm.nih.gov/38412345/"
      }
    ]
  }]
}
```

### Step 2: PubMed Fetch by PMIDs (Optional)

For fetching specific papers by their PubMed IDs:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "pubmed_fetch", "pmids": ["38412345", "38412346"]}'
```

Or with comma-separated string:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "pubmed_fetch", "pmids": "38412345,38412346"}'
```

### Step 3: bioRxiv Fetch (Date Range)

Fetch recent preprints from bioRxiv:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "biorxiv_fetch", "days": 30}'
```

Or with specific date range:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "biorxiv_fetch", "start_date": "2024-01-01", "end_date": "2024-02-01"}'
```

Response:
```json
{
  "task": "literature_search",
  "query_type": "biorxiv_fetch",
  "results": [{
    "start_date": "2024-01-01",
    "end_date": "2024-02-01",
    "papers_found": 500,
    "papers": [
      {
        "title": "Novel mechanism of cancer resistance",
        "authors": "Author list",
        "abstract": "Full abstract...",
        "doi": "10.1101/2024.01.01.12345",
        "date": "2024-01-01",
        "category": "cancer_biology",
        "link": "https://www.biorxiv.org/content/10.1101/2024.01.01.12345"
      }
    ]
  }]
}
```

### Step 4: bioRxiv Category Filter

Filter bioRxiv papers by subject category:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "biorxiv_category", "category": "immunology", "days": 30}'
```

## Expected Outputs

### PubMed Results

| Field | Description |
|-------|-------------|
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Full abstract |
| `doi` | DOI identifier |
| `pmid` | PubMed ID |
| `date` | Publication date |
| `journal` | Journal name |
| `link` | PubMed URL |

### bioRxiv Results

| Field | Description |
|-------|-------------|
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Full abstract |
| `doi` | DOI identifier |
| `date` | Publication date |
| `category` | Subject category |
| `link` | bioRxiv URL |

## Category Filters for bioRxiv

| Category | Description |
|----------|-------------|
| `cancer_biology` | Cancer research |
| `immunology` | Immune system studies |
| `cell_biology` | Cellular processes |
| `bioinformatics` | Computational biology |
| `neuroscience` | Nervous system research |
| `microbiology` | Microbial studies |
| `genomics` | Genome analysis |

## Example Usage

**Input**: "Find recent papers on CRISPR gene editing in cancer"

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "pubmed_search", "query": "CRISPR gene editing cancer", "max_results": 5}'
```

**Input**: "Get recent immunology preprints from bioRxiv"

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "biorxiv_category", "category": "immunology", "days": 30}'
```

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz`). If unreachable, re-resolve the base URL.

### No PubMed Results

**Symptom**: `papers_found: 0`.

**Solution**: Broaden search terms, check spelling, try alternative keywords.

### bioRxiv Timeout

**Symptom**: Request takes too long or fails.

**Solution**: Reduce date range (use smaller `days` value), retry.

### Empty Abstract

**Symptom**: Paper returns `"abstract": "No abstract available"`.

**Solution**: Some papers may not have abstracts indexed. Check the full paper via the link.

### Rate Limiting

**Symptom**: Multiple requests fail in sequence.

**Solution**: NCBI recommends max 3 requests/second. Add delay between requests.

## Limitations

- **PubMed**: Keyword search via NCBI E-utilities API
- **bioRxiv**: Date-range or category-based fetch only - no direct keyword search
- bioRxiv API returns all papers in date range (may be large)
- For comprehensive search, combine PubMed keyword search + bioRxiv category fetch

## API References

- **PubMed E-utilities**: https://www.ncbi.nlm.nih.gov/books/NBK25500/
- **bioRxiv API**: https://api.biorxiv.org/