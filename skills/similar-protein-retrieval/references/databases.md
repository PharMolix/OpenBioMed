# Databases Reference

This document describes the databases used for protein similarity search.

## Input Databases

### UniProt (Universal Protein Resource)

**URL**: https://www.uniprot.org/

**Description**: Comprehensive resource for protein sequence and annotation data.

**When used**: When user provides a UniProt ID (e.g., P0DTC2).

**Data retrieved**:
- Protein sequence
- Protein name and description
- Organism information
- Gene name
- PDB cross-references (structures)
- Functional annotations

**API endpoint**: `https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json`

### PDB (Protein Data Bank)

**URL**: https://www.rcsb.org/

**Description**: Archive of 3D structural data for biological macromolecules.

**When used**: When user provides a PDB ID (e.g., 6LZG).

**Data retrieved**:
- 3D coordinates (PDB file)
- Structure metadata
- Experimental method (X-ray, EM, NMR)
- Resolution

**API endpoint**: `https://files.rcsb.org/download/{pdb_id}.pdb`

### AlphaFold DB

**URL**: https://alphafold.ebi.ac.uk/

**Description**: Predicted protein structures using AlphaFold AI system.

**When used**: For proteins without experimental structures.

**Coverage**: 200+ million protein structures.

**API endpoint**: `https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}`

---

## Search Databases

### UniRef (for MSA)

**Description**: Clustered sets of UniProt sequences at different identity thresholds.

**Clusters**:
| Database | Identity Threshold | Size |
|----------|-------------------|------|
| UniRef100 | 100% | ~250M sequences |
| UniRef90 | 90% | ~60M sequences |
| UniRef50 | 50% | ~30M sequences |

**When used**: MSA sequence similarity search via ColabFold API.

### PDB100 (for FoldSeek)

**Description**: All PDB structures clustered at 100% sequence identity.

**Size**: ~200,000 structures

**When used**: FoldSeek structure similarity search.

### AFDB50 (for FoldSeek)

**Description**: AlphaFold DB structures clustered at 50% sequence identity.

**Size**: ~2 million structures

**When used**: FoldSeek structure similarity search for predicted structures.

### BFVD (Big Fantastic Virus Database)

**Description**: Predicted viral protein structures.

**When used**: FoldSeek search for viral proteins.

---

## Output Interpretation

### MSA Output (.a3m format)

A3M format is a multiple sequence alignment format used by HH-suite.

**Structure**:
```
>sequence_id_1
ACDEFGHIKLMNPQRSTVWY-
>sequence_id_2
ACD-FGHIKLMNPQRSTVWYZ
```

**Key features**:
- Lowercase letters: insertions (not aligned)
- Uppercase letters: aligned positions
- Dashes (-): gaps

### FoldSeek Output (.m8 format)

Tab-separated format similar to BLAST output.

**Columns**:
| Column | Name | Description |
|--------|------|-------------|
| 1 | query | Query sequence ID |
| 2 | target | Target sequence/structure ID |
| 3 | identity | Sequence identity (%) |
| 4 | alignment_length | Length of alignment |
| 5 | mismatch | Number of mismatches |
| 6 | gap_open | Number of gap openings |
| 7 | query_start | Start position in query |
| 8 | query_end | End position in query |
| 9 | target_start | Start position in target |
| 10 | target_end | End position in target |
| 11 | prob | Probability score |
| 12 | evalue | E-value |
| 13+ | Additional | Aligned sequences, coordinates |

---

## Rate Limits and Performance

| Service | Rate Limit | Typical Response Time |
|---------|------------|----------------------|
| UniProt API | 10 requests/sec | < 1 second |
| PDB Download | No formal limit | < 5 seconds |
| ColabFold MSA | 5 requests/second | 1-2 minutes |
| FoldSeek | 5 requests/second | 30 seconds - 2 minutes |

**Tips**:
- MSA takes longer for longer sequences
- FoldSeek is faster for smaller structures
- Both services have built-in retry logic
