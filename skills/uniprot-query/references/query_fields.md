# UniProt Query Fields Reference

Complete list of query fields for searching UniProtKB.

## Basic Query Syntax

```
field:value
field:value1 OR field:value2
field1:value1 AND field2:value2
```

## Gene and Protein

| Field | Example | Description |
|-------|---------|-------------|
| `gene` | `gene:BRCA` | Gene name (partial match) |
| `gene_exact` | `gene_exact:EGFR` | Exact gene name match |
| `gene_synonym` | `gene_synonym:HER1` | Gene synonym |
| `protein_name` | `protein_name:kinase` | Protein name |
| `protein_full_name` | `protein_full_name:"Epidermal growth factor receptor"` | Full protein name |
| `ec` | `ec:2.7.10.1` | Enzyme Commission number |

## Organism

| Field | Example | Description |
|-------|---------|-------------|
| `organism_id` | `organism_id:9606` | NCBI Taxonomy ID |
| `organism` | `organism:"Homo sapiens"` | Organism name |
| `host_id` | `host_id:9606` | Host organism (for viral proteins) |
| `host_name` | `host_name:"Homo sapiens"` | Host organism name |

### Common Organism IDs

| Organism | Taxon ID |
|----------|----------|
| Human (Homo sapiens) | 9606 |
| Mouse (Mus musculus) | 10090 |
| Rat (Rattus norvegicus) | 10116 |
| Zebrafish (Danio rerio) | 7955 |
| Fruit fly (Drosophila melanogaster) | 7227 |
| Nematode (C. elegans) | 6239 |
| Yeast (S. cerevisiae) | 559292 |
| E. coli K-12 | 83333 |
| Arabidopsis thaliana | 3702 |
| SARS-CoV-2 | 2697049 |
| SARS-CoV | 694009 |
| HIV-1 | 11676 |

## Sequence Properties

| Field | Example | Description |
|-------|---------|-------------|
| `length` | `length:[100 TO 500]` | Sequence length range |
| `mass` | `mass:[10000 TO 50000]` | Molecular mass (Da) |
| `sequence` | `sequence:MKFLILLFNI` | Sequence search |
| `checksum` | `checksum:A836172E9D` | Sequence CRC64 checksum |

## Annotation

| Field | Example | Description |
|-------|---------|-------------|
| `keyword` | `keyword:Kinase` | UniProt keyword |
| `keyword_id` | `keyword_id:KW-0418` | Keyword ID |
| `cc_function` | `cc_function:apoptosis` | Function comment |
| `cc_disease` | `cc_disease:diabetes` | Disease annotation |
| `cc_subcellular_location` | `cc_subcellular_location:membrane` | Subcellular location |
| `cc_tissue_specificity` | `cc_tissue_specificity:liver` | Tissue specificity |
| `domain` | `domain:kinase` | Domain annotation |
| `family` | `family:"protein kinase"` | Protein family |

## Post-Translational Modifications

| Field | Example | Description |
|-------|---------|-------------|
| `ft_mod_res` | `ft_mod_res:"Phosphoserine"` | Modified residue |
| `ft_carbohyd` | `ft_carbohyd:true` | Glycosylation site |
| `ft_lipid` | `ft_lipid:true` | Lipidation site |
| `ft_disulfid` | `ft_disulfid:true` | Disulfide bond |

## Structural Features

| Field | Example | Description |
|-------|---------|-------------|
| `ft_transmem` | `ft_transmem:true` | Transmembrane region |
| `ft_signal` | `ft_signal:true` | Signal peptide |
| `ft_helix` | `ft_helix:true` | Helical region |
| `ft_strand` | `ft_strand:true` | Beta strand |
| `ft_turn` | `ft_turn:true` | Turn region |

## Entry Properties

| Field | Example | Description |
|-------|---------|-------------|
| `reviewed` | `reviewed:true` | Swiss-Prot (reviewed) only |
| `created` | `created:[2020-01-01 TO *]` | Entry creation date |
| `modified` | `modified:[2023-01-01 TO *]` | Last modified date |
| `version` | `version:123` | Entry version number |
| `annotation_score` | `annotation_score:5` | Annotation quality (1-5) |

## Cross-References

| Field | Example | Description |
|-------|---------|-------------|
| `database:(type:PDB)` | `database:(type:PDB)` | Has PDB entry |
| `database:(type:PDB AND 1ABC)` | Specific PDB ID |
| `xref_pdb` | `xref_pdb:1ABC` | PDB cross-reference |
| `xref_hgnc` | `xref_hgnc:HGNC:3236` | HGNC ID |
| `xref_ensembl` | `xref_ensembl:ENSP00000275493` | Ensembl ID |
| `xref_refseq` | `xref_refseq:NP_005219` | RefSeq ID |

## Query Examples

### Find all human kinases
```
keyword:Kinase AND organism_id:9606 AND reviewed:true
```

### Find SARS-CoV-2 proteins
```
organism_id:2697049
```

### Find proteins with transmembrane domains
```
ft_transmem:true AND organism_id:9606
```

### Find proteins involved in diabetes
```
cc_disease:diabetes AND organism_id:9606
```

### Find secreted proteins in human
```
ft_signal:true AND cc_subcellular_location:secreted AND organism_id:9606
```

### Find membrane proteins between 300-500 aa
```
ft_transmem:true AND length:[300 TO 500] AND organism_id:9606
```

## Operators

| Operator | Example | Description |
|----------|---------|-------------|
| AND | `gene:BRCA AND organism_id:9606` | Both conditions |
| OR | `gene:EGFR OR gene:HER2` | Either condition |
| NOT | `kinase NOT receptor` | Exclude |
| * | `gene:BRCA*` | Wildcard |
| [x TO y] | `length:[100 TO 500]` | Range |

## References

- UniProt Query Syntax: https://www.uniprot.org/help/text-search
- Query Fields: https://www.uniprot.org/help/query-fields
