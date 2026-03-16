---
name: pubchem-query
description: >
  Query PubChem database for chemical structures, similar compounds, and bioactivity data.
  Use this skill when:
  (1) Converting drug name to molecular structure (SMILES, SDF),
  (2) Finding similar compounds for lead optimization,
  (3) Querying bioactivity data against protein targets,
  (4) Getting compounds active in specific assays.
license: MIT
category: data-retrieval
tags: [pubchem, compound-search, bioactivity, similarity-search]
---

# PubChem Query

Query PubChem database for drug discovery and chemistry applications.

## When to Use

- Convert drug name to molecular structure (SMILES, SDF)
- Find similar compounds for lead optimization
- Query bioactivity data against protein targets
- Get compounds active in specific assays

## Workflow

### Use Case 1: Name/ID to Structure

```python
from open_biomed.tools.tool_registry import TOOLS

tool = TOOLS["molecule_name_request"]
molecules, _ = tool.run("aspirin")
mol = molecules[0]
print(f"SMILES: {mol.smiles}")
```

### Use Case 2: Similarity Search

```python
from open_biomed.data import Molecule

query = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
tool = TOOLS["molecule_structure_request"]
molecules, _ = tool.run(molecule=query, threshold=0.85, max_records=10)
for mol in molecules:
    print(mol.smiles)
```

### Use Case 3: Bioactivity Query

```python
tool = TOOLS["pubchem_bioactivity"]

# Query 1: Get assays where compound was active
results, _ = tool.run(query_type="compound", cid=2244, aids_type="active")

# Query 2: Get compounds active in an assay
results, _ = tool.run(query_type="assay", aid=1195, cids_type="active")

# Query 3: Get assays targeting a gene
results, _ = tool.run(query_type="target", gene_symbol="PTGS2")
```

## Expected Outputs

| Query Type | Output |
|------------|--------|
| Name to Structure | `Molecule` object with SMILES, SDF file saved |
| Similarity Search | List of similar `Molecule` objects |
| Bioactivity (compound) | List of AIDs where compound was active/inactive |
| Bioactivity (assay) | List of CIDs active/inactive in the assay |
| Bioactivity (target) | List of AIDs targeting the gene |

## Score Interpretation

| Similarity Threshold | Interpretation |
|---------------------|----------------|
| > 0.90 | Very similar, likely same scaffold |
| 0.80-0.90 | Similar, potential analogs |
| 0.70-0.80 | Moderately similar, scaffold hops possible |

## Error Handling

| Error | Solution |
|-------|----------|
| Compound not found | Try alternative names or SMILES |
| No similar compounds | Lower threshold (min 0.70) |
| No bioactivity data | Compound may not be tested; try related compounds |
| Timeout | Reduce max_records or retry |

## Available Tools

| Tool Name | Purpose |
|-----------|---------|
| `molecule_name_request` | Name/CID to structure |
| `molecule_structure_request` | Similarity search |
| `pubchem_bioactivity` | Bioactivity queries |

See `examples/basic_example.py` for complete runnable examples.
