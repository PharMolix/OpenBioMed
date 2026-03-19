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

Query the KEGG (Kyoto Encyclopedia of Genes and Genomes) database for comprehensive biomedical information.

## When to Use

- **Drug Lookup**: Retrieve drug efficacy, targets, metabolism, drug-drug interactions
- **Pathway Analysis**: Get pathway genes, compounds, modules, and related pathways
- **Disease Discovery**: Find disease-associated drugs, genes, and therapeutic targets

## Workflow

### Use Case 1: Drug Information Lookup

Fetch comprehensive drug information from KEGG DRUG database.

```python
from scripts.kegg_api import kegg_find, kegg_get, parse_drug_entry

# Step 1: Search for drug by name
results = kegg_find("drug", "aspirin")
# Returns: [("dr:D00109", "Aspirin (JP18/USP); Acetylsalicylic acid; ...")]

# Step 2: Get full entry
drug_id = "dr:D00109"  # or just "D00109"
entry = kegg_get(drug_id)
drug_info = parse_drug_entry(entry)
```

**Output includes**: Names, formula, efficacy, diseases, targets, pathways, metabolism, DDI.

See `examples/drug_lookup.py` for complete implementation.

### Use Case 2: Pathway Analysis

Analyze KEGG pathways to retrieve genes, compounds, and modules.

```python
from scripts.kegg_api import kegg_get, parse_pathway_entry

# Get pathway by ID (e.g., hsa00010 for Glycolysis)
entry = kegg_get("hsa00010")
pathway = parse_pathway_entry(entry)

# Access parsed data
print(f"Genes: {len(pathway['genes'])}")      # 50+ genes
print(f"Compounds: {len(pathway['compounds'])}")  # 30+ compounds
```

**Output includes**: Description, genes with KO/EC annotations, compounds, modules, related pathways.

See `examples/pathway_analysis.py` for complete implementation.

### Use Case 3: Disease-Drug-Target Discovery

Discover therapeutic targets and drugs for diseases.

```python
from scripts.kegg_api import kegg_find, kegg_get, parse_disease_entry

# Step 1: Search for disease
results = kegg_find("disease", "diabetes")
# Returns multiple matches including Type 2 diabetes (H00409)

# Step 2: Get disease details
entry = kegg_get("ds:H00409")
disease = parse_disease_entry(entry)

# Access drugs and targets
print(f"Drugs: {len(disease['drugs'])}")    # 60+ drugs
print(f"Genes: {len(disease['genes'])}")    # 20+ genes
```

**Output includes**: Description, category, associated genes, pathways, approved drugs.

See `examples/disease_discovery.py` for complete implementation.

## Expected Outputs

### Drug Entry (JSON)

```json
{
  "id": "D00109",
  "names": ["Aspirin", "Acetylsalicylic acid"],
  "formula": "C9H8O4",
  "efficacy": ["Analgesic", "Anti-inflammatory", "Antipyretic", "COX inhibitor"],
  "targets": [
    {"gene": "PTGS1", "uniprot": "P23219", "ko": "K00509"},
    {"gene": "PTGS2", "uniprot": "P35354", "ko": "K11987"}
  ],
  "pathways": ["hsa00590", "hsa04611"],
  "diseases": ["Myocardial infarction", "Unstable angina"]
}
```

### Pathway Entry (JSON)

```json
{
  "id": "hsa00010",
  "name": "Glycolysis / Gluconeogenesis",
  "organism": "Homo sapiens",
  "description": "Glycolysis is the process...",
  "genes": [
    {"id": "10327", "symbol": "AKR1A1", "ko": "K00002", "ec": "1.1.1.2"},
    {"id": "3939", "symbol": "LDHA", "ko": "K00016", "ec": "1.1.1.27"}
  ],
  "compounds": [
    {"id": "C00031", "name": "D-Glucose"},
    {"id": "C00022", "name": "Pyruvate"}
  ],
  "modules": ["hsa_M00001", "hsa_M00002", "hsa_M00003"]
}
```

### Disease Entry (JSON)

```json
{
  "id": "H00409",
  "name": "Type 2 diabetes mellitus",
  "category": "Endocrine and metabolic disease",
  "description": "T2DM is characterized by chronic hyperglycemia...",
  "genes": [
    {"symbol": "CAPN10", "ko": "K08579"},
    {"symbol": "TCF7L2", "ko": "K04491"}
  ],
  "drugs": [
    {"id": "D00944", "name": "Metformin hydrochloride"},
    {"id": "D06404", "name": "Liraglutide"}
  ],
  "pathways": ["hsa04930", "hsa04911"]
}
```

## KEGG API Reference

| Operation | URL Pattern | Description |
|-----------|-------------|-------------|
| `info` | `/info/{database}` | Database statistics |
| `list` | `/list/{database}` | List all entries |
| `find` | `/find/{database}/{query}` | Search by keyword |
| `get` | `/get/{entry_id}` | Retrieve entry |
| `link` | `/link/{target}/{source}` | Cross-references |
| `conv` | `/conv/{target}/{source}` | ID conversion |

**Key Databases**: `pathway`, `compound`, `drug`, `disease`, `genes`, `enzyme`, `ko`

**Entry ID Formats**:
- Drug: `D00009` or `dr:D00009`
- Compound: `C00031` or `cpd:C00031`
- Pathway: `hsa00010` (organism-specific) or `map00010` (reference)
- Disease: `H00409` or `ds:H00409`
- Gene: `hsa:5742` (organism:gene_id)

## Error Handling

| Error | Solution |
|-------|----------|
| Entry not found | Verify ID format (e.g., D00109, not aspirin) |
| Multiple matches | Use `kegg_find` first to get exact ID |
| Timeout | Reduce query complexity, retry with delay |
| Rate limited | KEGG allows ~10 requests/second; add delays |

## Integration with OpenBioMed

```python
from open_biomed.data import Molecule, Protein
from open_biomed.tools.tool_registry import TOOLS

# Convert KEGG compound to Molecule
compound_entry = kegg_get("cpd:C00031")  # Glucose
mol_file = kegg_get("C00031", option="mol")  # Get MOL format
# molecule = Molecule.from_mol_file(mol_file)

# Get protein from KEGG gene
gene_entry = kegg_get("hsa:5742")  # PTGS1
# Use UniProt ID to fetch protein
protein_tool = TOOLS["protein_uniprot_request"]
proteins, _ = protein_tool.run(accession="P23219")
```

## References

- `references/kegg_databases.md` - Complete database listing and ID formats
- `references/kegg_api_operations.md` - Detailed API operation reference
- KEGG API Documentation: https://www.kegg.jp/kegg/rest/keggapi.html
