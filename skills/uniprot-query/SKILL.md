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

Query the UniProt knowledgebase for comprehensive protein information.

## When to Use

- Look up protein by UniProt accession (e.g., P00533 for EGFR)
- Search proteins by gene name, organism, or keywords
- Retrieve protein metadata: function, domains, diseases, PTMs
- Get protein sequences and structural annotations

## Workflow

### Use Case 1: Protein Lookup by ID

Fetch complete protein information including metadata.

```python
from open_biomed.tools.tool_registry import TOOLS
import requests
import json

# Get protein sequence (existing tool)
tool = TOOLS["protein_uniprot_request"]
proteins, _ = tool.run(accession="P0DTC2")  # SARS-CoV-2 Spike
protein = proteins[0]

# Fetch full metadata from UniProt API
url = f"https://rest.uniprot.org/uniprotkb/P0DTC2?format=json"
response = requests.get(url)
metadata = parse_uniprot_entry(response.json())
```

See `examples/lookup_by_id.py` for complete implementation.

### Use Case 2: Search by Criteria

Search UniProt by gene name, organism, keywords, or disease.

```python
import requests

base_url = "https://rest.uniprot.org/uniprotkb/search"

# Example queries:
queries = {
    "gene_exact:EGFR AND organism_id:9606": "Human EGFR",
    "gene_exact:S AND organism_id:2697049": "SARS-CoV-2 Spike",
    "keyword:Kinase AND organism_id:9606": "Human kinases",
    "diabetes AND organism_id:9606": "Diabetes-related proteins",
}

params = {
    "query": "gene_exact:EGFR AND organism_id:9606 AND reviewed:true",
    "fields": "accession,gene_primary,protein_name,organism_name,length",
    "format": "json",
    "size": 10
}
response = requests.get(base_url, params=params)
```

See `examples/search_by_criteria.py` for complete implementation.

## Query Syntax Reference

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

**Combine queries**: Use `AND`, `OR` to combine criteria:
- `gene_exact:EGFR AND organism_id:9606 AND reviewed:true`

## Expected Outputs

### Metadata JSON (lookup_by_id)

```json
{
  "accession": "P0DTC2",
  "uniProtId": "SPIKE_SARS2",
  "protein": {"name": "Spike glycoprotein"},
  "gene": {"primary": "S", "synonyms": []},
  "organism": {"scientific_name": "...", "taxon_id": 2697049},
  "sequence": {"length": 1273, "mass": 141178},
  "function": ["Attaches the virion to host receptor..."],
  "domains": [{"type": "Domain", "description": "RBD", "location": "319-541"}],
  "keywords": ["Glycoprotein", "Transmembrane", "Viral attachment"],
  "subcellular_location": ["Virion membrane"]
}
```

### Text Report

```
======================================================================
UNIPROT PROTEIN REPORT
======================================================================
Accession:     P0DTC2
Protein:       Spike glycoprotein
Gene:          S
Organism:      Severe acute respiratory syndrome coronavirus 2
Length:        1273 aa

FUNCTION
Attaches the virion to the cell membrane by interacting with
host receptor ACE2...

DOMAINS
• Domain: BetaCoV S1-NTD (14-303)
• Region: Receptor-binding domain (319-541)
```

### Search Results JSON

```json
{
  "query": "gene_exact:S AND organism_id:2697049",
  "total_results": 10,
  "results": [
    {"accession": "P0DTC2", "gene": "S", "protein_name": "Spike glycoprotein", ...}
  ]
}
```

## Error Handling

| Error | Solution |
|-------|----------|
| Accession not found | Verify UniProt ID format (e.g., P00533, not EGFR) |
| No search results | Broaden query, remove `reviewed:true`, check organism ID |
| Timeout | Reduce `size` parameter, simplify query |
| Rate limited | Wait and retry; UniProt allows 10 requests/second |

## Available Tools

| Tool | Purpose |
|------|---------|
| `protein_uniprot_request` | Fetch protein sequence by accession (existing) |

The workflows in this skill extend the basic tool with full metadata retrieval via UniProt REST API.

## References

- `references/query_fields.md` - Complete query field reference
- `references/metadata_fields.md` - Available metadata fields
- UniProt API Docs: https://www.uniprot.org/api-documentation
