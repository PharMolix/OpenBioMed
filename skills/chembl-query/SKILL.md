---
name: chembl-query
description: >
  Query ChEMBL database for bioactivity data on drug-like compounds.
  Use this skill when:
  (1) Finding compounds active against a protein target (target-based search),
  (2) Getting bioactivity profile for a molecule (molecule-based search),
  (3) Finding drugs for a disease indication (indication-based search).
license: MIT
category: knowledge-retrieval
tags: [chembl, bioactivity, drug-discovery, target, indication]
---

# ChEMBL Query

Query ChEMBL database for bioactivity data on drug-like compounds.

## When to Use

- Find compounds active against a protein target (target-based search)
- Get bioactivity profile for a molecule (molecule-based search)
- Find drugs for a disease indication (indication-based search)

## Workflow

### Use Case 1: Target-Based Compound Search

Find compounds with activity against a protein target.

```python
from open_biomed.tools.tool_registry import TOOLS

tool = TOOLS["chembl_query"]

# Search by target name
results, _ = tool.run(
    query_type="target",
    target_name="EGFR",
    standard_type="IC50",
    standard_value_lte=100,  # nM
    limit=20
)

# Or search by UniProt ID
results, _ = tool.run(
    query_type="target",
    uniprot_id="P00533",
    standard_type="IC50",
    limit=20
)
```

### Use Case 2: Molecule Bioactivity Profile

Get all known targets and activity data for a compound.

```python
# Search by molecule name
results, _ = tool.run(
    query_type="molecule",
    molecule_name="imatinib",
    limit=50
)

# Or search by SMILES
results, _ = tool.run(
    query_type="molecule",
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    limit=20
)

# Or search by ChEMBL ID
results, _ = tool.run(
    query_type="molecule",
    chembl_id="CHEMBL25",
    limit=20
)
```

### Use Case 3: Disease/Indication-Based Drug Search

Find drugs studied for a specific disease.

```python
# Find all drugs for diabetes
results, _ = tool.run(
    query_type="indication",
    disease="diabetes",
    limit=50
)

# Filter for approved drugs only (max_phase=4)
results, _ = tool.run(
    query_type="indication",
    disease="diabetes",
    max_phase=4,  # Approved drugs only
    limit=20
)
```

## Expected Outputs

| Query Type | Output Fields |
|------------|---------------|
| Target | `molecule_chembl_id`, `molecule_name`, `target_chembl_id`, `target_name`, `standard_type`, `standard_value`, `standard_units`, `pchembl_value` |
| Molecule | `molecule_chembl_id`, `molecule_name`, `target_chembl_id`, `target_name`, `target_organism`, `standard_type`, `standard_value`, `standard_units`, `pchembl_value` |
| Indication | `molecule_chembl_id`, `molecule_name`, `indication`, `max_phase_for_ind`, `phase_description` |

## Score Interpretation

### Activity Values

| pChEMBL Value | IC50/Ki Approx | Interpretation |
|---------------|----------------|----------------|
| > 9 | < 1 nM | Extremely potent |
| 8-9 | 1-10 nM | Very potent |
| 7-8 | 10-100 nM | Potent |
| 6-7 | 100 nM - 1 uM | Moderately active |
| 5-6 | 1-10 uM | Weakly active |
| < 5 | > 10 uM | Inactive |

### Clinical Phase

| Phase | Description |
|-------|-------------|
| 0 | Preclinical |
| 1 | Phase I (safety) |
| 2 | Phase II (efficacy) |
| 3 | Phase III (large-scale) |
| 4 | Approved |

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query_type` | str | "target", "molecule", or "indication" |
| `target_name` | str | Target name (e.g., "EGFR", "BACE1") |
| `uniprot_id` | str | UniProt accession (e.g., "P00533") |
| `molecule_name` | str | Molecule name (e.g., "aspirin") |
| `smiles` | str | SMILES string |
| `chembl_id` | str | ChEMBL ID (e.g., "CHEMBL25") |
| `disease` | str | Disease name (e.g., "diabetes") |
| `standard_type` | str | Activity type (e.g., "IC50", "Ki", "EC50") |
| `standard_value_lte` | float | Max activity value (nM) |
| `max_phase` | int | Minimum clinical phase (0-4) |
| `limit` | int | Max results (default: 50) |

## Error Handling

| Error | Solution |
|-------|----------|
| Target not found | Try alternative names or UniProt ID |
| No activity data | Target may not have curated bioactivity data |
| Molecule not found | Verify SMILES or try molecule name |
| No indication results | Try simpler disease terms (e.g., "neoplasm" instead of "cancer") |
| Timeout | Reduce limit parameter |

## Available Tools

| Tool Name | Purpose |
|-----------|---------|
| `chembl_query` | Unified ChEMBL query interface |

See `examples/basic_example.py` for complete runnable examples.