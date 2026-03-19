# KEGG API Operations Reference

Detailed reference for KEGG REST API operations.

## Base URL

```
https://rest.kegg.jp
```

## Operations Overview

| Operation | URL Pattern | Description |
|-----------|-------------|-------------|
| `info` | `/info/{database}` | Database statistics |
| `list` | `/list/{database}` | List entries |
| `find` | `/find/{database}/{query}` | Search entries |
| `get` | `/get/{entry_id}` | Retrieve entry |
| `link` | `/link/{target}/{source}` | Cross-references |
| `conv` | `/conv/{target}/{source}` | ID conversion |
| `ddi` | `/ddi/{drug_id}` | Drug-drug interactions |

---

## info - Database Statistics

Retrieve database release information and entry counts.

### Syntax
```
/info/{database}
/info/{organism_code}
```

### Examples
```bash
# Get KEGG release statistics
curl "https://rest.kegg.jp/info/kegg"

# Get pathway database statistics
curl "https://rest.kegg.jp/info/pathway"

# Get human genome statistics
curl "https://rest.kegg.jp/info/hsa"
```

### Output
```
kegg  Kyoto Encyclopedia of Genes and Genomes
      Kanehisa Laboratories
      Release 113.0+/03-19, March 25
      530 pathways, 9392 modules, 25168 KO groups
      ...
```

---

## list - List Entries

List entry identifiers and names from a database.

### Syntax
```
/list/{database}
/list/{database}/{organism}
/list/{entry_ids}
```

### Examples
```bash
# List all reference pathways
curl "https://rest.kegg.jp/list/pathway"

# List all human pathways
curl "https://rest.kegg.jp/list/pathway/hsa"

# List all human genes
curl "https://rest.kegg.jp/list/hsa"

# List specific entries
curl "https://rest.kegg.jp/list/cpd:C00031+cpd:C00002"
```

### Output Format
```
path:map00010	Glycolysis / Gluconeogenesis
path:map00020	Citrate cycle (TCA cycle)
path:map00030	Pentose phosphate pathway
```

---

## find - Search Entries

Search for entries by keywords, formula, or molecular weight.

### Syntax
```
/find/{database}/{query}
/find/{database}/{query}/{option}
```

### Options (for compound/drug databases)
- `formula` - Chemical formula search
- `exact_mass` - Exact mass range
- `mol_weight` - Molecular weight range

### Examples
```bash
# Keyword search
curl "https://rest.kegg.jp/find/drug/aspirin"
curl "https://rest.kegg.jp/find/disease/diabetes"
curl "https://rest.kegg.jp/find/genes/p53"

# Chemical formula search
curl "https://rest.kegg.jp/find/compound/C6H12O6/formula"

# Exact mass range search
curl "https://rest.kegg.jp/find/compound/180-185/exact_mass"

# Molecular weight range search
curl "https://rest.kegg.jp/find/drug/300-310/mol_weight"
```

### Output Format
```
dr:D00109	Aspirin (JP18/USP); Acetylsalicylic acid
dr:D02079	Codein phosphate and aspirin
```

---

## get - Retrieve Entries

Retrieve complete database entries or specific data formats.

### Syntax
```
/get/{entry_id}
/get/{entry_id1}+{entry_id2}+...
/get/{entry_id}/{option}
```

### Options
| Option | Description | Databases |
|--------|-------------|-----------|
| `aaseq` | Amino acid sequence | genes |
| `ntseq` | Nucleotide sequence | genes |
| `mol` | MOL format | compound, drug, glycan |
| `kcf` | KCF format | compound, drug, glycan |
| `image` | PNG image | pathway, compound, drug |
| `kgml` | KGML (pathway XML) | pathway |

### Examples
```bash
# Get single entry
curl "https://rest.kegg.jp/get/dr:D00109"

# Get multiple entries (max 10)
curl "https://rest.kegg.jp/get/cpd:C00031+cpd:C00002+cpd:C00022"

# Get amino acid sequence
curl "https://rest.kegg.jp/get/hsa:10458/aaseq"

# Get compound structure (MOL format)
curl "https://rest.kegg.jp/get/cpd:C00031/mol"

# Get pathway image
curl "https://rest.kegg.jp/get/hsa00010/image" > pathway.png

# Get pathway KGML
curl "https://rest.kegg.jp/get/hsa00010/kgml" > pathway.xml
```

### Entry Limits
- Up to 10 entries per request
- Only 1 pathway entry for image/kgml option
- Only 1 compound/drug/glycan entry for image/mol option

---

## link - Cross-References

Find related entries between databases.

### Syntax
```
/link/{target_database}/{source_database}
/link/{target_database}/{source_entry}
```

### Examples
```bash
# Find all compounds in a pathway
curl "https://rest.kegg.jp/link/cpd/hsa00010"

# Find all pathways containing a compound
curl "https://rest.kegg.jp/link/pathway/cpd:C00031"

# Find all genes in a pathway
curl "https://rest.kegg.jp/link/hsa/hsa00010"

# Find all pathways containing a gene
curl "https://rest.kegg.jp/link/pathway/hsa:10458"

# Find all KOs for a gene
curl "https://rest.kegg.jp/link/ko/hsa:10458"

# Find all genes with a KO
curl "https://rest.kegg.jp/link/hsa/ko:K00001"
```

### Output Format
```
path:hsa00010	cpd:C00031
path:hsa00010	cpd:C00033
path:hsa00010	cpd:C00022
```

---

## conv - ID Conversion

Convert between KEGG IDs and external database IDs.

### Syntax
```
/conv/{target_database}/{source_database}
/conv/{target_database}/{source_entry}
```

### Supported External Databases

| Type | Database | ID Format |
|------|----------|-----------|
| Gene | `ncbi-geneid` | 10458 |
| Gene | `ncbi-proteinid` | NP_001258 |
| Gene | `uniprot` | P00533 |
| Chemical | `pubchem` | 2244 |
| Chemical | `chebi` | 15365 |
| Drug | `atc` | A01AD05 |

### Examples
```bash
# Convert UniProt to KEGG
curl "https://rest.kegg.jp/conv/hsa/uniprot:P00533"

# Convert KEGG gene to UniProt
curl "https://rest.kegg.jp/conv/uniprot/hsa:10458"

# Convert PubChem to KEGG compound
curl "https://rest.kegg.jp/conv/cpd/pubchem:2244"

# Convert KEGG compound to PubChem
curl "https://rest.kegg.jp/conv/pubchem/cpd:C00031"

# Convert all human genes to NCBI Gene IDs
curl "https://rest.kegg.jp/conv/ncbi-geneid/hsa"
```

### Output Format
```
hsa:10458	ncbi-geneid:10458
hsa:10459	ncbi-geneid:10459
```

---

## ddi - Drug-Drug Interactions

Retrieve drug-drug interaction information.

### Syntax
```
/ddi/{drug_id}
```

### Examples
```bash
# Get interactions for aspirin
curl "https://rest.kegg.jp/ddi/D00109"

# Get interactions for multiple drugs
curl "https://rest.kegg.jp/ddi/D00109+D00564+D00100"
```

### Output Format
```
dr:D00100	dr:D00564	CI,P	unclassified
dr:D00109	dr:D00564	P	unclassified
```

### Interaction Types
| Code | Description |
|------|-------------|
| `CI` | Contraindicated |
| `P` | Precaution |
| `C` | Caution |

---

## Python Implementation

```python
import requests
from typing import List, Tuple, Optional

KEGG_API_BASE = "https://rest.kegg.jp"

def kegg_request(operation: str, *args, **kwargs) -> str:
    """Make a KEGG API request."""
    url = f"{KEGG_API_BASE}/{operation}"
    if args:
        url += "/" + "/".join(str(a) for a in args)
    response = requests.get(url, **kwargs)
    response.raise_for_status()
    return response.text

def kegg_info(database: str) -> str:
    """Get database statistics."""
    return kegg_request("info", database)

def kegg_list(database: str, organism: Optional[str] = None) -> str:
    """List entries in a database."""
    if organism:
        return kegg_request("list", database, organism)
    return kegg_request("list", database)

def kegg_find(database: str, query: str, option: Optional[str] = None) -> str:
    """Search entries in a database."""
    if option:
        return kegg_request("find", database, query, option)
    return kegg_request("find", database, query)

def kegg_get(entry_ids: List[str], option: Optional[str] = None) -> str:
    """Retrieve entries by ID."""
    ids = "+".join(entry_ids)
    if option:
        return kegg_request("get", ids, option)
    return kegg_request("get", ids)

def kegg_link(target: str, source: str) -> str:
    """Find cross-references."""
    return kegg_request("link", target, source)

def kegg_conv(target: str, source: str) -> str:
    """Convert IDs between databases."""
    return kegg_request("conv", target, source)

def kegg_ddi(drug_ids: List[str]) -> str:
    """Get drug-drug interactions."""
    ids = "+".join(drug_ids)
    return kegg_request("ddi", ids)
```

---

## Rate Limiting

- Maximum: ~10 requests per second
- For bulk operations, add delays between requests
- Consider using batch operations (get up to 10 entries at once)

```python
import time

def batch_get(entry_ids: List[str], batch_size: int = 10, delay: float = 0.2):
    """Batch retrieve entries with rate limiting."""
    results = []
    for i in range(0, len(entry_ids), batch_size):
        batch = entry_ids[i:i + batch_size]
        result = kegg_get(batch)
        results.append(result)
        time.sleep(delay)
    return results
```

---

## Error Handling

| HTTP Status | Meaning | Solution |
|-------------|---------|----------|
| 200 | Success | - |
| 400 | Bad request | Check URL format |
| 404 | Not found | Entry doesn't exist |
| 403 | Forbidden | Rate limit exceeded |

```python
def safe_kegg_request(url: str, max_retries: int = 3):
    """Make request with retry logic."""
    import time
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```
