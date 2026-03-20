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

Query the STRING database to retrieve protein-protein interactions with comprehensive confidence scores.

## When to Use

- Find interaction partners for a protein (by UniProt ID)
- Retrieve confidence scores for PPIs (experimental, text mining, database)
- Build protein interaction networks for pathway analysis
- Identify potential protein complexes or functional modules

## Workflow

### Basic Query

```python
from open_biomed.tools.tool_registry import TOOLS

# Query STRING for interaction partners
tool = TOOLS["ppi_string_request"]
results, _ = tool.run(uniprot_id="P04637")  # TP53

# Access results
for interaction in results:
    print(f"{interaction['partner_gene']}: {interaction['combined_score']}")
```

### Custom Parameters

```python
# High confidence interactions only, limit to 20
results, _ = tool.run(
    uniprot_id="P04637",
    species=9606,           # Human (default)
    required_score=700,     # High confidence (default)
    limit=20                # Max interactors
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uniprot_id` | str | required | UniProt accession (e.g., P04637) |
| `species` | int | 9606 | NCBI taxonomy ID (9606=human) |
| `required_score` | int | 700 | Min confidence (150/400/700/900) |
| `limit` | int | 50 | Max interactors to return |

## Confidence Score Thresholds

| Score | Level | Use Case |
|-------|-------|----------|
| 150 | Low | Exploratory analysis |
| 400 | Medium | Balanced retrieval |
| 700 | High | Reliable interactions (default) |
| 900 | Highest | Very confident only |

## Expected Output

```json
[
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
```

## Score Interpretation

| Score Type | Source | Range |
|------------|--------|-------|
| `combined_score` | Weighted combination | 0-1 |
| `experimental` | Wet-lab experiments | 0-1 |
| `text_mining` | Literature co-occurrence | 0-1 |
| `database` | Curated databases (BioGRID, etc.) | 0-1 |
| `coexpression` | Expression correlation | 0-1 |
| `phylogenetic` | Phylogenetic profiles | 0-1 |
| `gene_fusion` | Fusion events | 0-1 |
| `neighborhood` | Genomic proximity | 0-1 |

## Error Handling

| Error | Solution |
|-------|----------|
| No interactions found | Lower `required_score` threshold |
| UniProt ID not recognized | Verify ID format (e.g., P04637) |
| Rate limited | Wait and retry; STRING allows ~5 req/sec |
| Wrong species | Check NCBI taxonomy ID |

## Common Organism IDs

| Organism | Taxonomy ID |
|----------|-------------|
| Human | 9606 |
| Mouse | 10090 |
| Rat | 10116 |
| E. coli | 83333 |
| S. cerevisiae | 4932 |

## References

- `examples/basic_query.py` - Complete example script
- `references/score_details.md` - Detailed score methodology
- STRING API Docs: https://string-db.org/help/api/
