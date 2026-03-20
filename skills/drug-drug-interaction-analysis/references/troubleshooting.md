# Troubleshooting Guide

## Common Issues and Solutions

### 1. Drug Not Found

**Problem**: Drug name returns no KEGG ID.

**Solutions**:
- Try alternative drug names (generic vs brand)
- Use PubChem to find SMILES, then search KEGG by structure
- Check spelling and special characters

```python
# Alternative approach using PubChem
from open_biomed.tools.tool_registry import TOOLS

tool = TOOLS['molecule_name_request']
result, _ = tool.run(accession='tylenol')  # Brand name for acetaminophen
```

### 2. No Interactions Found

**Problem**: Query returns empty results.

**Possible causes**:
- Drugs have no known interactions (good news!)
- Drugs are not in KEGG database
- Using compound IDs instead of drug IDs

**Solution**: Check if drug exists in KEGG Drug database:
```python
# Verify drug is in KEGG
curl "https://rest.kegg.jp/find/drug/YOUR_DRUG_NAME"
```

### 3. API Rate Limiting

**Problem**: 403 Forbidden errors.

**Solution**: Add delay between requests:
```python
import time

def batch_ddi(drug_pairs, delay=0.5):
    results = []
    for pair in drug_pairs:
        result = get_interactions(pair)
        results.append(result)
        time.sleep(delay)
    return results
```

### 4. Timeout Errors

**Problem**: Request times out.

**Solutions**:
- Increase timeout value
- Retry with exponential backoff
- Reduce number of drugs per query

```python
import time

def retry_request(url, max_retries=3, timeout=60):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            return response
        except requests.Timeout:
            wait = 2 ** attempt
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

### 5. Unexpected Severity Codes

**Problem**: Severity code not recognized.

**Solution**: KEGG may use combined codes (e.g., "CI,P" for both Contraindicated and Precaution):
```python
def parse_severity(code):
    codes = code.split(',')
    severity_map = {
        "CI": "Contraindicated",
        "P": "Precaution",
        "C": "Caution"
    }
    return [severity_map.get(c, c) for c in codes]
```

## KEGG DDI Response Format

### Standard Response
```
dr:D00109	dr:D00126	P	Target: PTGS1 PTGS2
```

| Field | Description |
|-------|-------------|
| Column 1 | First drug KEGG ID |
| Column 2 | Second drug KEGG ID |
| Column 3 | Severity code (CI/P/C) |
| Column 4 | Mechanism description |

### Severity Codes

| Code | Full Name | Clinical Action |
|------|-----------|-----------------|
| **CI** | Contraindicated | Do not use together |
| **P** | Precaution | Monitor closely; consider alternatives |
| **C** | Caution | Be aware; observe patient |

## Alternative Data Sources

If KEGG DDI doesn't have the interaction:

1. **DrugBank** (requires license)
   - https://go.drugbank.com/drug_drug_interaction_checker

2. **Way2Drug DDI-Pred** (free prediction)
   - https://way2drug.com/ddi/
   - Accepts SMILES for novel compounds

3. **DDInter 2.0** (free database)
   - https://ddinter2.scbdd.com/
   - 302K DDI records

4. **Medscape Drug Interaction Checker** (free web)
   - https://reference.medscape.com/drug-interactionchecker

## Integration with OpenBioMed

```python
from open_biomed.tools.tool_registry import TOOLS

# Get molecule from name
tool = TOOLS['molecule_name_request']
molecule, _ = tool.run(accession='aspirin')
smiles = molecule[0].smiles

# Use SMILES for DDI prediction (if drug not in KEGG)
# Then query Way2Drug DDI-Pred
```
