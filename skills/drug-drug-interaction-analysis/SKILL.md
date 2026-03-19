---
name: drug-drug-interaction-analysis
description: >
  Analyze potential drug-drug interactions (DDI) for up to 5 drugs using KEGG DDI database.
  Use this skill when:
  (1) Checking interactions between multiple medications,
  (2) Assessing DDI risk for drug combinations,
  (3) Understanding interaction mechanisms and severity,
  (4) Analyzing CYP enzyme involvement in DDIs.
license: MIT
category: safety-toxicology
tags: [ddi, drug-interaction, pharmacology, clinical-decision, cyp]
---

# Drug-Drug Interaction Analysis

Analyze potential drug-drug interactions (DDI) for medication safety assessment.

## When to Use

- Checking interactions between prescribed medications
- Evaluating DDI risk for polypharmacy patients
- Understanding interaction mechanisms (CYP enzymes, shared targets)
- Clinical decision support for drug combinations

## Workflow

### Step 1: Resolve Drug Names to KEGG IDs

```python
import requests

KEGG_API = "https://rest.kegg.jp"

def find_drug_id(drug_name: str) -> str:
    """Find KEGG drug ID from drug name."""
    response = requests.get(f"{KEGG_API}/find/drug/{drug_name}")
    if response.ok and response.text.strip():
        # Parse first result: "dr:D00109\tAspirin..."
        line = response.text.strip().split('\n')[0]
        return line.split('\t')[0]  # Returns "dr:D00109"
    return None
```

### Step 2: Query KEGG DDI API

```python
def get_ddi(drug_ids: list) -> list:
    """Query KEGG DDI for multiple drugs."""
    ids = "+".join(drug_ids)
    response = requests.get(f"{KEGG_API}/ddi/{ids}")
    interactions = []
    for line in response.text.strip().split('\n'):
        if line:
            parts = line.split('\t')
            interactions.append({
                "drug_a": parts[0],
                "drug_b": parts[1],
                "severity": parts[2],
                "mechanism": parts[3] if len(parts) > 3 else ""
            })
    return interactions
```

### Step 3: Get Detailed Drug Information

```python
def get_drug_info(drug_id: str) -> dict:
    """Get detailed drug information from KEGG."""
    response = requests.get(f"{KEGG_API}/get/{drug_id}")
    info = {"id": drug_id, "targets": [], "enzymes": []}
    for line in response.text.split('\n'):
        if line.startswith("NAME"):
            info["name"] = line.split(maxsplit=1)[1].strip()
        elif line.startswith("TARGET"):
            info["targets"].append(line.split(maxsplit=1)[1])
        elif line.startswith("METABOLISM"):
            info["enzymes"].append(line.split(maxsplit=1)[1])
    return info
```

### Step 4: Analyze and Report

```python
def analyze_ddi(drugs: list) -> dict:
    """Full DDI analysis workflow."""
    # Resolve drug IDs
    drug_ids = [find_drug_id(d) for d in drugs]

    # Get interactions
    interactions = get_ddi([d.replace("dr:", "") for d in drug_ids if d])

    # Get drug details for context
    drug_info = {d: get_drug_info(d) for d in drug_ids if d}

    return {
        "drugs_analyzed": drugs,
        "drug_ids": drug_ids,
        "interactions": interactions,
        "drug_details": drug_info
    }
```

## Expected Outputs

### Interaction Result Structure

```json
{
  "drugs_analyzed": ["aspirin", "ibuprofen"],
  "total_pairs": 1,
  "interactions_found": 1,
  "results": [
    {
      "drug_a": "Aspirin (D00109)",
      "drug_b": "Ibuprofen (D00126)",
      "severity": "Precaution (P)",
      "mechanism": "Shared target: PTGS1 PTGS2 (COX enzymes)",
      "clinical_note": "Ibuprofen may interfere with aspirin's antiplatelet effect"
    }
  ]
}
```

### Severity Levels

| Code | Severity | Description |
|------|----------|-------------|
| **CI** | Contraindicated | Should not be used together |
| **P** | Precaution | Monitor closely; adjust if needed |
| **C** | Caution | Be aware; may need intervention |

### Mechanism Types

- **Target overlap**: Both drugs act on same protein targets
- **CYP interaction**: One drug affects metabolism of another
- **Pharmacodynamic**: Additive or opposing effects
- **Pharmacokinetic**: Absorption/distribution/excretion effects

## Error Handling

1. **Drug not found**: Try alternative names or SMILES lookup via PubChem
2. **No interactions**: May indicate no known DDIs or drugs not in database
3. **API timeout**: Retry with exponential backoff; max 10 requests/second
4. **Invalid input**: Validate drug names/SMILES before processing

## Limitations

- KEGG DDI covers approved drugs only
- Novel compounds require predictive models (Way2Drug DDI-Pred)
- Severity classifications are database-defined
- Always consult clinical resources for patient-specific decisions

## References

- KEGG DDI API: https://rest.kegg.jp/ddi/
- KEGG Drug Database: https://www.kegg.jp/kegg/drug/
