# UniProt Metadata Fields Reference

Fields available in UniProt protein entries.

## Entry Identification

| Field | JSON Path | Description |
|-------|-----------|-------------|
| Accession | `primaryAccession` | Primary UniProt accession (e.g., P00533) |
| Entry Name | `uniProtkbId` | Entry name (e.g., EGFR_HUMAN) |
| Entry Type | `entryType` | Swiss-Prot or TrEMBL |
| Secondary Accessions | `secondaryAccessions` | Historical accessions |

## Protein Description

| Field | JSON Path | Description |
|-------|-----------|-------------|
| Recommended Name | `proteinDescription.recommendedName.fullName.value` | Primary protein name |
| Short Name | `proteinDescription.recommendedName.shortName` | Abbreviated name |
| EC Numbers | `proteinDescription.recommendedName.ecNumbers` | Enzyme Commission numbers |
| Alternative Names | `proteinDescription.alternativeNames` | Alternative protein names |
| Flags | `proteinDescription.flag` | e.g., "Precursor" |

## Gene Information

| Field | JSON Path | Description |
|-------|-----------|-------------|
| Primary Gene Name | `genes[0].geneName.value` | Primary gene symbol |
| Gene Synonyms | `genes[0].synonyms` | Alternative gene names |
| ORF Names | `genes[0].orfNames` | Open reading frame names |

## Organism

| Field | JSON Path | Description |
|-------|-----------|-------------|
| Scientific Name | `organism.scientificName` | Scientific name |
| Common Name | `organism.commonName` | Common name |
| Taxon ID | `organism.taxonId` | NCBI Taxonomy ID |
| Lineage | `organism.lineage` | Taxonomic lineage |

## Sequence

| Field | JSON Path | Description |
|-------|-----------|-------------|
| Sequence | `sequence.value` | Amino acid sequence |
| Length | `sequence.length` | Sequence length (aa) |
| Mass | `sequence.molWeight` | Molecular weight (Da) |
| CRC64 Checksum | `sequence.crc64Checksum` | Sequence checksum |
| Modified | `sequence.modified` | Last modification date |

## Comments (Annotations)

### Function
```json
{
  "commentType": "FUNCTION",
  "texts": [{"value": "Protein function description..."}]
}
```

### Disease
```json
{
  "commentType": "DISEASE",
  "disease": {
    "diseaseId": "Lung cancer",
    "acronym": "LNCR",
    "description": "Disease description...",
    "diseaseCrossReference": {"database": "MIM", "id": "211980"}
  }
}
```

### Subcellular Location
```json
{
  "commentType": "SUBCELLULAR LOCATION",
  "subcellularLocations": [
    {"location": {"value": "Cell membrane"}}
  ]
}
```

### Other Comment Types

| Type | Description |
|------|-------------|
| CATALYTIC ACTIVITY | Enzyme reactions |
| COFACTOR | Required cofactors |
| ACTIVITY REGULATION | Regulation of activity |
| SUBUNIT | Subunit structure |
| INTERACTION | Protein interactions |
| TISSUE SPECIFICITY | Tissue expression |
| DEVELOPMENTAL STAGE | Developmental expression |
| INDUCTION | Induction conditions |
| PTM | Post-translational modifications |
| ALTERNATIVE PRODUCTS | Isoforms |
| PHARMACOLOGICAL | Drug target info |
| MISCELLANEOUS | Other annotations |
| SIMILARITY | Sequence similarities |
| CAUTION | Important notes |

## Features (Sequence Annotations)

| Type | Description |
|------|-------------|
| Domain | Protein domain |
| Repeat | Repeated sequence |
| Region | Functional region |
| Motif | Short motif |
| Active site | Catalytic residue |
| Binding site | Ligand binding |
| Site | Other site |
| Transmembrane | Transmembrane region |
| Intramembrane | Intramembrane region |
| Signal | Signal peptide |
| Peptide | Processed peptide |
| Propeptide | Propeptide region |
| Chain | Mature chain |
| Cross-link | Cross-link |
| Disulfide bond | Disulfide bond |
| Glycosylation | Glycosylation site |
| Lipidation | Lipid attachment |
| Modified residue | Other modification |
| Natural variant | Natural variant |
| Mutagenesis | Mutagenesis site |
| Sequence conflict | Sequence conflict |
| Sequence uncertainty | Uncertain region |
| Helix | Secondary structure |
| Strand | Secondary structure |
| Turn | Secondary structure |

### Feature JSON Structure
```json
{
  "type": "Domain",
  "description": "Receptor-binding domain",
  "location": {
    "start": {"value": 319, "modifier": "EXACT"},
    "end": {"value": 541, "modifier": "EXACT"}
  }
}
```

## Keywords

Keywords categorize proteins by function, PTM, subcellular location, etc.

```json
{
  "keywords": [
    {"name": "3D-structure", "id": "KW-0001"},
    {"name": "Glycoprotein", "id": "KW-0325"}
  ]
}
```

### Common Keyword Categories

| Category | Examples |
|----------|----------|
| Molecular function | Kinase, Receptor, Transferase |
| Biological process | Apoptosis, Cell cycle, DNA repair |
| Cellular component | Membrane, Nucleus, Cytoplasm |
| PTM | Phosphoprotein, Glycoprotein, Acetylation |
| Technical term | 3D-structure, Reference proteome |

## Cross-References

External database links in `uniProtKBCrossReferences`:

| Database | Description |
|----------|-------------|
| PDB | 3D structures |
| AlphaFoldDB | Predicted structures |
| HGNC | Human gene nomenclature |
| Ensembl | Genome database |
| RefSeq | NCBI reference sequences |
| GeneID | NCBI Gene |
| KEGG | Pathway database |
| GO | Gene Ontology |
| Pfam | Protein families |
| InterPro | Integrated protein signatures |
| SMART | Protein domains |
| PROSITE | Protein patterns |
| DrugBank | Drug database |
| ChEMBL | Bioactive molecules |
| ClinVar | Clinical variants |
| dbSNP | SNP database |

### Cross-Reference JSON Structure
```json
{
  "database": "PDB",
  "id": "1IVO",
  "properties": [
    {"key": "Method", "value": "X-ray"},
    {"key": "Resolution", "value": "2.80 Å"}
  ]
}
```

## References

Literature citations in `references`:
- Citation details (authors, title, journal)
- PubMed ID
- DOI
- Evidence mapping to annotations

## Extra Attributes

| Field | Description |
|-------|-------------|
| `annotationScore` | Overall annotation quality (1-5) |
| `uniParcId` | UniParc cross-reference |
| `proteinExistence` | Evidence for protein existence |

## JSON API Response Structure

```
{
  "entryType": "UniProtKB reviewed (Swiss-Prot)",
  "primaryAccession": "P00533",
  "uniProtkbId": "EGFR_HUMAN",
  "organism": {...},
  "proteinDescription": {...},
  "genes": [...],
  "comments": [...],
  "features": [...],
  "keywords": [...],
  "references": [...],
  "uniProtKBCrossReferences": [...],
  "sequence": {...},
  "extraAttributes": {...}
}
```
