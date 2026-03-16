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
category: knowledge-retrieval
tags: [literature-search, pubmed, biorxiv, preprints, research-papers]
---

# Biomedical Literature Search

Search PubMed and bioRxiv for biomedical research papers with titles and abstracts.

## When to Use

- Find research papers on a specific biomedical topic
- Retrieve recent preprints from bioRxiv
- Get paper titles, abstracts, authors, and links
- Literature review for drug discovery or biomedical research

## Workflow

### PubMed Search (Keyword-based)

```python
import requests
import xml.etree.ElementTree as ET

# Step 1: Search for PMIDs
search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {"db": "pubmed", "term": "PD-1 inhibitor cancer", "retmax": 10, "retmode": "json"}
response = requests.get(search_url, params=params)
pmids = response.json()["esearchresult"]["idlist"]

# Step 2: Fetch paper details
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
response = requests.get(fetch_url, params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"})
```

### bioRxiv Fetch (Date-range based)

```python
import requests

# Fetch papers by date range
url = "https://api.biorxiv.org/details/biorxiv/2026-02-01/2026-03-01"
response = requests.get(url)
papers = response.json()["collection"]

for paper in papers[:5]:
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract'][:200]}...")
```

## Expected Outputs

### PubMed Results
Returns list of papers with:
| Field | Description |
|-------|-------------|
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Full abstract |
| `doi` | DOI identifier |
| `pmid` | PubMed ID |
| `date` | Publication date |
| `link` | PubMed URL |

### bioRxiv Results
Returns list of papers with:
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

## Error Handling

| Error | Solution |
|-------|----------|
| No PubMed results | Broaden search terms, check spelling |
| bioRxiv timeout | Reduce date range, retry |
| Empty abstract | Paper may not have abstract available |
| Rate limiting | Add delay between requests (NCBI: 3 req/sec) |

## API References

- **PubMed E-utilities**: https://www.ncbi.nlm.nih.gov/books/NBK25500/
- **bioRxiv API**: https://api.biorxiv.org/

## Notes

- **PubMed**: Keyword search via NCBI E-utilities API
- **bioRxiv**: Date-range or category-based fetch via bioRxiv API
- bioRxiv does not support direct keyword search
- For comprehensive search, use both sources together

See `examples/basic_example.py` for complete runnable examples.
