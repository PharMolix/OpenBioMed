---
name: structure-prediction-boltz-2
description: >
  Structure prediction using Boltz-2, an open biomolecular structure predictor.
  Use this skill when:
  (1) Predicting protein complex structures,
  (2) Validating designed binders,
  (3) Predicting protein-ligand complexes and binding affinity,
  (4) External GPU resources (no local model weights required).

  For protein complex binding affinity evaluation, use prodigy.
license: MIT
category: design-tools
tags: [structure-prediction, protein complex, protein-ligand complex, affinity]
---

# Boltz-2 Structure Prediction

Predict protein complex structures and protein-ligand affinity using Boltz-2 external API via the `/run_pipeline/` endpoint.

## API Endpoints

**OpenBioMed Pipeline API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/`

Environment variable override:
- `BOLTZ2_API_BASE_URL`: Override Boltz-2 API base URL
- `PIPELINE_API_URL`: Override OpenBioMed pipeline API URL

## Execution Flow

1. **Submit Job**: Send request to Boltz-2 API (returns job_id immediately)
2. **Poll Status**: Wait for job completion (submit + poll pattern)
3. **Fetch Result**: Retrieve structure and affinity data when completed
4. **Save and Display**: Files saved to `./tmp/boltz2/`

## Task Types

| Task | Required Inputs | Output |
|------|-----------------|--------|
| affinity | sequence, smiles | PDB file + IC50 value |
| prot_complex | sequence_1, sequence_2 | PDB file |

## Step 1: Protein-Ligand Affinity Prediction

### Input Collection
- `task`: `"boltz2_structure_prediction"` (OpenBioMed task name)
- `prediction_type`: `"affinity"` (prediction mode)
- `sequence`: Protein amino acid sequence (required)
- `smiles`: Ligand SMILES string (required)
- `task_id`: Project/batch ID (optional, auto-generated)
- `task_name`: Task name (optional, auto-generated)
- `output_name`: Output file name prefix (optional)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "boltz2_structure_prediction",
    "prediction_type": "affinity",
    "sequence": "GSHMGSSGMSSGMG",
    "smiles": "CCO",
    "output_name": "my_affinity_pred"
  }'
```

### Response

```json
{
  "task": "boltz2_structure_prediction",
  "prediction_type": "affinity",
  "output_files": [
    "./tmp/boltz2/my_affinity_pred.pdb",
    "./tmp/boltz2/my_affinity_pred_affinity.json",
    "./tmp/boltz2/my_affinity_pred_result.json"
  ],
  "description": "Boltz-2 Affinity prediction completed.\nJob ID: 83573f9c-bd2a-...\nStructure: ./tmp/boltz2/my_affinity_pred.pdb\nAffinity prediction: 0.6200\nIC50: 7.34 nM"
}
```

## Step 2: Protein Complex Structure Prediction

### Input Collection
- `task`: `"boltz2_structure_prediction"` (OpenBioMed task name)
- `prediction_type`: `"prot_complex"` (prediction mode)
- `sequence_1`: First protein sequence (chain A, required)
- `sequence_2`: Second protein sequence (chain B, required)
- `task_id`: Project/batch ID (optional)
- `task_name`: Task name (optional)
- `output_name`: Output file name prefix (optional)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "boltz2_structure_prediction",
    "prediction_type": "prot_complex",
    "sequence_1": "GSHMGSSGMSSGMG",
    "sequence_2": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    "output_name": "my_complex"
  }'
```

### Response

```json
{
  "task": "boltz2_structure_prediction",
  "prediction_type": "prot_complex",
  "output_files": [
    "./tmp/boltz2/my_complex.pdb",
    "./tmp/boltz2/my_complex_result.json"
  ],
  "description": "Boltz-2 Prot-complex prediction completed.\nJob ID: 4a961520-...\nStructure: ./tmp/boltz2/my_complex.pdb\nStructure length: 150 residues"
}
```

## Output Interpretation

### Generated Files

| File Type | Content |
|-----------|---------|
| `.pdb` | Predicted 3D structure |
| `_affinity.json` | Affinity prediction (for affinity mode) |
| `_result.json` | Complete job metadata |

### Affinity JSON Structure

```json
{
  "affinity": 0.6200282573699951,
  "ic50": 7.338281456947327
}
```

- `affinity`: Raw affinity prediction value (higher = stronger binding)
- `ic50`: Converted IC50 value in nM (lower = stronger binding)

## Polling Behavior

The Boltz-2 API uses a submit + poll pattern:
- **Submit**: Returns job_id immediately
- **Poll**: Wait for status = "completed" (default 10s interval, 600s timeout)
- **MSA Search**: Typically takes 30s-3min (cached for repeated sequences)
- **Structure Prediction**: Typically takes 20s-5min

## Error Handling

### Missing Required Parameters

**Symptom**: Error message about missing inputs

**Solution**: Ensure required parameters for prediction type:
- affinity: sequence + smiles
- prot_complex: sequence_1 + sequence_2

### Job Timeout

**Symptom**: Request takes very long (>10 minutes)

**Solution**: MSA search may be running. Wait and retry, or check job status via API.

### Job Failed

**Symptom**: API returns status "failed"

**Solution**: Check error message in response. Common causes:
- Invalid sequence format
- MSA search failed
- GPU memory exhausted

## Sequence Format Requirements

- Plain amino acid string (no FASTA header)
- Standard 20 amino acid codes (ACDEFGHIKLMNPQRSTVWY)
- No spaces or special characters

## Decision Tree

```
What task type?
│
├─ Protein-ligand affinity → prediction_type: "affinity"
│   └─ Provide sequence + smiles
│   └─ Returns structure + IC50
│
└─ Protein complex structure → prediction_type: "prot_complex"
    └─ Provide sequence_1 + sequence_2
    └─ Returns structure (two chains)
```

## Typical Performance

| Campaign Size | Time | Notes |
|---------------|------|-------|
| 1 affinity prediction | 1-3 min | Includes MSA search |
| 10 complexes | 10-30 min | Batch processing |
| 100 complexes | 30-90 min | Parallel queue |

**Per-complex**: ~20s-5min depending on sequence length and MSA complexity.

## Next Steps

After structure prediction:
1. **Binding Affinity**: For protein complexes, use `binding-affinity-prediction-prodigy`
2. **Visualization**: Open PDB in PyMol or molecular viewer
3. **Analysis**: Examine interface residues and contacts