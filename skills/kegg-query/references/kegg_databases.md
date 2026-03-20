# KEGG Databases Reference

Complete listing of KEGG databases and their identifier formats.

## Core Databases

| Database | Prefix | ID Format | Description |
|----------|--------|-----------|-------------|
| Pathway | `path`/`map` | `map00010`, `hsa00010` | Metabolic and signaling pathways |
| Brite | `br` | `br08303` | Hierarchical classifications |
| Module | `M` | `M00001` | Functional modules |
| KO (KEGG Orthology) | `ko`/`K` | `K00001` | Ortholog groups |
| Enzyme | `ec` | `ec:1.1.1.1` | Enzyme Commission numbers |

## Gene Databases

| Database | Prefix | ID Format | Description |
|----------|--------|-----------|-------------|
| Genes | - | `hsa:10458` | Organism-specific genes |
| Genome | `T` | `T01001` | KEGG organisms |
| VG (Virus) | `vg` | `vg:12345` | Viral genomes |
| VP (Viral Protein) | `vp` | `vp:12345` | Viral proteins |

## Chemical Databases

| Database | Prefix | ID Format | Description |
|----------|--------|-----------|-------------|
| Compound | `cpd`/`C` | `C00031` | Small molecules |
| Glycan | `gl`/`G` | `G00001` | Glycans/polysaccharides |
| Drug | `dr`/`D` | `D00109` | Approved drugs |
| Dgroup | `DG` | `DG00015` | Drug groups |

## Reaction Databases

| Database | Prefix | ID Format | Description |
|----------|--------|-----------|-------------|
| Reaction | `rn`/`R` | `R00001` | Biochemical reactions |
| Rclass | `RC` | `RC00001` | Reaction classes |

## Health Databases

| Database | Prefix | ID Format | Description |
|----------|--------|-----------|-------------|
| Disease | `ds`/`H` | `H00409` | Human diseases |
| Network | `nt` | `nt06549` | Disease networks |
| Variant | - | - | Genetic variants |

## Organism Codes

Common KEGG organism codes for pathway queries:

| Code | Organism | Taxon ID |
|------|----------|----------|
| `hsa` | Homo sapiens (human) | 9606 |
| `mmu` | Mus musculus (mouse) | 10090 |
| `rno` | Rattus norvegicus (rat) | 10116 |
| `dre` | Danio rerio (zebrafish) | 7955 |
| `dme` | Drosophila melanogaster (fruit fly) | 7227 |
| `cel` | Caenorhabditis elegans (nematode) | 6239 |
| `sce` | Saccharomyces cerevisiae (yeast) | 559292 |
| `eco` | Escherichia coli K-12 | 83333 |
| `ath` | Arabidopsis thaliana | 3702 |

### Pathway ID Convention

- **Reference pathways**: `map#####` (generic, organism-independent)
- **Organism-specific**: `<org>#####` (e.g., `hsa00010` for human Glycolysis)

Example:
- `map00010` = Reference Glycolysis pathway
- `hsa00010` = Human Glycolysis pathway
- `mmu00010` = Mouse Glycolysis pathway

## External Database Mappings

KEGG supports ID conversion to/from external databases:

### Gene IDs
- `ncbi-geneid` - NCBI Gene
- `ncbi-proteinid` - NCBI Protein (RefSeq)
- `uniprot` - UniProt
- `pubmed` - PubMed

### Chemical IDs
- `pubchem` - PubChem CID
- `chebi` - ChEBI
- `atc` - ATC Classification
- `jtc` - Japanese Therapeutic Category

## Entry Format Examples

### Drug Entry (D00109 - Aspirin)
```
ENTRY       D00109                      Drug
NAME        Aspirin (JP18/USP); Acetylsalicylic acid
FORMULA     C9H8O4
EXACT_MASS  180.0423
MOL_WEIGHT  180.16
EFFICACY    Analgesic, Anti-inflammatory, Antipyretic
TARGET      PTGS1 (COX1) [HSA:5742] [KO:K00509]
PATHWAY     hsa00590(5742+5743)  Arachidonic acid metabolism
```

### Compound Entry (C00031 - Glucose)
```
ENTRY       C00031                      Compound
NAME        D-Glucose; D-Glucopyranose
FORMULA     C6H12O6
EXACT_MASS  180.0634
MOL_WEIGHT  180.16
```

### Pathway Entry (hsa00010 - Glycolysis)
```
ENTRY       hsa00010                    Pathway
NAME        Glycolysis / Gluconeogenesis - Homo sapiens (human)
GENE        10327  AKR1A1; aldo-keto reductase [KO:K00002] [EC:1.1.1.2]
COMPOUND    C00022  Pyruvate
MODULE      hsa_M00001  Glycolysis (Embden-Meyerhof pathway)
```

### Disease Entry (H00409 - Type 2 Diabetes)
```
ENTRY       H00409                      Disease
NAME        Type 2 diabetes mellitus
CATEGORY    Endocrine and metabolic disease
GENE        (T2D1) CAPN10 [HSA:11132] [KO:K08579]
DRUG        Metformin hydrochloride [DR:D00944]
PATHWAY     hsa04930  Type II diabetes mellitus
```

## Rate Limits

- Maximum requests: ~10 requests/second
- Entries per request (get): Up to 10 entries
- Use delays between requests for large queries
