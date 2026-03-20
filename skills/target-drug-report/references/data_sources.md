# Data Sources Reference

## Primary Data Sources

### 1. ClinicalTrials.gov
- **URL**: https://clinicaltrials.gov
- **Content**: Clinical trial registry and results database
- **Access**: Web search or API v2
- **Data Available**:
  - Study titles and descriptions
  - Intervention/drug names
  - Phase information
  - Recruitment status
  - Sponsor information

### 2. PubMed
- **URL**: https://pubmed.ncbi.nlm.nih.gov
- **Content**: Biomedical literature database
- **Access**: Web search or NCBI E-utilities API
- **Data Available**:
  - Research paper abstracts
  - Publication dates
  - Author information
  - MeSH terms

### 3. UniProt
- **URL**: https://www.uniprot.org
- **Content**: Protein sequence and functional information
- **Access**: REST API
- **Data Available**:
  - Protein sequences
  - Functional annotations
  - Disease associations
  - Post-translational modifications

### 4. ChEMBL (Future Integration)
- **URL**: https://www.ebi.ac.uk/chembl
- **Content**: Bioactive molecule database
- **Access**: REST API
- **Data Available**:
  - Compound structures
  - Bioactivity data
  - Target associations
  - Drug indications

## Search Query Templates

### Clinical Trial Search
```
{target_name} inhibitor clinical trial phase {phase}
{target_name} inhibitor site:clinicaltrials.gov
{target_name} antagonist recruiting trial
```

### Paper Search
```
{target_name} inhibitor mechanism site:pubmed.ncbi.nlm.nih.gov
{target_name} drug discovery 2024 2025
{target_name} targeted therapy review
```

### Drug Approval Search
```
{target_name} inhibitor FDA approved
{target_name} antagonist approved drug
{target_name} inhibitor regulatory approval
```

## Known Target UniProt IDs

| Target | UniProt ID | Disease Area |
|--------|------------|--------------|
| EGFR | P00533 | Oncology |
| KRAS | P01116 | Oncology |
| BCL-2 | P10415 | Oncology |
| ALK | Q9UM73 | Oncology |
| BRAF | P15056 | Oncology |
| HER2 | P04626 | Oncology |
| PD-1 | Q15116 | Immunology |
| CTLA-4 | P16410 | Immunology |
| JAK2 | O60674 | Hematology |
| BTK | Q06187 | Hematology |

## Rate Limits and Best Practices

1. **Web Search**: Add delays between queries to avoid rate limiting
2. **API Calls**: Respect API rate limits (usually 3 requests/second)
3. **Caching**: Cache results for repeated queries
4. **Error Handling**: Implement exponential backoff for failures
