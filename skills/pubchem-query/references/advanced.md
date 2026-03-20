# Advanced PubChem Query Usage

## API Endpoints

The skill uses PubChem PUG REST API:

| Query Type | Endpoint |
|------------|----------|
| Name to Structure | `compound/name/{name}/SDF` |
| CID to Structure | `compound/cid/{cid}/SDF` |
| Similarity Search | `compound/fastsimilarity_2d/smiles/{smiles}/cids/JSON` |
| Compound Bioactivity | `compound/cid/{cid}/aids/XML?aids_type={type}` |
| Assay Compounds | `assay/aid/{aid}/cids/XML?cids_type={type}` |
| Target Assays | `assay/target/genesymbol/{gene}/aids/JSON` |

## Query Parameters

### Similarity Search

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `threshold` | 0.0-1.0 | 0.8 | Tanimoto similarity threshold |
| `max_records` | 1-1000 | 10 | Maximum results to return |

### Bioactivity Query Types

| Parameter | Values | Description |
|-----------|--------|-------------|
| `aids_type` | all, active, inactive | Filter assays by activity |
| `cids_type` | all, active, inactive | Filter compounds by activity |

## Complete Workflow Example

### Drug Repurposing Analysis

```python
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.data import Molecule

# Step 1: Get structure of a drug
name_tool = TOOLS["molecule_name_request"]
molecules, _ = name_tool.run("imatinib")
drug = molecules[0]

# Step 2: Find similar compounds
sim_tool = TOOLS["molecule_structure_request"]
similar, _ = sim_tool.run(molecule=drug, threshold=0.85, max_records=20)

# Step 3: Get bioactivity for original drug
bio_tool = TOOLS["pubchem_bioactivity"]
assays, _ = bio_tool.run(query_type="compound", cid=5291, aids_type="active")

# Step 4: Get assays targeting a specific target
target_assays, _ = bio_tool.run(query_type="target", gene_symbol="ABL1")

# Step 5: Find intersection - compounds active in target assays
for aid in [r["AID"] for r in target_assays[:5]]:
    compounds, _ = bio_tool.run(query_type="assay", aid=aid, cids_type="active")
    print(f"Assay {aid}: {len(compounds)} active compounds")
```

## Rate Limiting

PubChem recommends max 5 requests per second. The tools include automatic rate limiting.

```python
# For bulk queries, add delays
import time
for cid in cids:
    results, _ = tool.run(query_type="compound", cid=cid)
    time.sleep(0.2)  # 5 requests per second max
```

## Common Gene Symbols for Drug Targets

| Gene | Target | Therapeutic Area |
|------|--------|------------------|
| PTGS2 | COX-2 | Inflammation, Pain |
| HMGCR | HMG-CoA reductase | Cholesterol |
| ACHE | Acetylcholinesterase | Alzheimer's |
| DRD2 | Dopamine D2 receptor | Psychiatry |
| EGFR | EGFR tyrosine kinase | Oncology |
| BACE1 | Beta-secretase 1 | Alzheimer's |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty results | Compound not in PubChem | Try alternative names or InChIKey |
| Timeout | Large result set | Reduce max_records parameter |
| XML parsing error | API change | Check PubChem API status |
| Rate limited | Too many requests | Add delays between calls |

## Integration with Other Tools

```python
from open_biomed.tools.tool_registry import TOOLS

# Get compound, then analyze drug-likeness
molecules, _ = TOOLS["molecule_name_request"].run("aspirin")
qed_tool = TOOLS["molecule_qed"]
qed_score, _ = qed_tool.run(molecules[0])
print(f"QED score: {qed_score}")
```
