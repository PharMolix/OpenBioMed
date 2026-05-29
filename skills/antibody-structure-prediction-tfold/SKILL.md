---
name: antibody-structure-prediction-tfold
description: >
  Antibody-related structure prediction using tFold model.
  Use this skill when:
  (1) Predict antibody and nanobody structure of a given sequence,
  (2) Predict antigen-antibody complex structure of given sequences,
  (3) Using local GPU resources.

  For binding affinity evaluation, use binding-affinity-prediction-prodigy.
license: MIT
category: design-tools
tags: [structure-prediction, antibody, nanobody, antigen-antibody complex, tfold]
---

# tFold Antibody-related Structure Prediction

Predict antibody structure and antigen-antibody complex structure using tFold deep learning model.

## When to Use

- User wants to predict antibody/nanobody structure from sequence
- User wants to predict antigen-antibody complex structure
- User provides heavy chain and light chain sequences (for antibody)
- User provides antigen + heavy chain + light chain (for complex)

## API Endpoint Resolution

The skill resolves the OpenBioMed API base URL in this order:

1. **Environment variable**: `${OPENBIOMED_API_BASE_URL}` (if set)
2. **Docker container default**: `http://openbiomed-server:8090` (if running in Docker)
3. **Local development default**: `http://127.0.0.1:8090`

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL.

## Workflow

### Step 1: Prepare Input Sequences

Collect FASTA sequences for:

| Mode | Required Inputs | Description |
|------|-----------------|-------------|
| `antibody` | heavy_chain + light_chain | Antibody/nanobody structure |
| `complex` | heavy_chain + light_chain + antigen | Antigen-antibody complex |

**Example sequences**:
```
Heavy chain: EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK

Light chain: DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK
```

### Step 2: Call antibody_structure API

#### Antibody Structure Prediction

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "antibody_structure", "heavy_chain": "<HEAVY_CHAIN>", "light_chain": "<LIGHT_CHAIN>"}'
```

**Response**:
```json
{
  "task": "antibody_structure",
  "mode": "antibody",
  "pdb_path": "./tmp/antibody_structure_xxx.pdb",
  "description": "Antibody structure predicted and saved to ./tmp/antibody_structure_xxx.pdb"
}
```

#### Antigen-Antibody Complex Prediction

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "antibody_structure", "heavy_chain": "<HEAVY_CHAIN>", "light_chain": "<LIGHT_CHAIN>", "antigen": "<ANTIGEN_SEQUENCE>", "mode": "complex"}'
```

**Response**:
```json
{
  "task": "antibody_structure",
  "mode": "complex",
  "pdb_path": "./tmp/antibody_structure_xxx.pdb",
  "description": "Complex structure predicted and saved to ./tmp/antibody_structure_xxx.pdb"
}
```

### Step 3: View and Use Results

The predicted structure is saved as a PDB file. You can:

1. **Visualize**: Use `visualize_protein` task or PyMol
2. **Analyze binding**: Use `binding_affinity` task to predict binding affinity
3. **Download**: Copy the PDB file from the server

## Example Usage

### Example 1: Predict Antibody Structure

```
Input: "Predict the structure of this antibody with heavy chain EVQL... and light chain DIQMT..."

Step 1: Prepare sequences
  Heavy chain: EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK
  Light chain: DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK

Step 2: Call API

  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "antibody_structure", "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK", "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK"}'

Output:
  PDB file: ./tmp/antibody_structure_xxx.pdb
  Contains predicted 3D structure of the antibody
```

## Expected Outputs

| Output | Type | Description |
|--------|------|-------------|
| pdb_path | string | Path to predicted PDB file |
| mode | string | "antibody" or "complex" |
| description | string | Human-readable description |

## Error Handling

### Missing Sequences

**Symptom**: API returns error about missing sequences.

**Solution**: Ensure both heavy_chain and light_chain are provided. For complex mode, also provide antigen.

### Model Loading Error

**Symptom**: API returns "tFold not installed" error.

**Solution**: tFold needs to be installed in the server environment:
```bash
pip install tfold termcolor deepspeed ml-collections dm-tree modelcif
```

### GPU Memory Error

**Symptom**: Prediction fails with CUDA out of memory.

**Solution**: tFold requires significant GPU memory (24GB+ recommended). Try:
- Using a GPU with more memory
- Reducing sequence length

## Decision Tree

```
Should I use tFold?
│
└─ What are you predicting?
   ├─ Antibody/nanobody structure → antibody-structure-prediction-tfold ✓
   ├─ Antigen-antibody complex → antibody-structure-prediction-tfold ✓
   ├─ General protein-protein complex → structure-prediction-boltz-2
   └─ Protein-ligand complex → structure-prediction-boltz-2
```

## Next Steps

After structure prediction:
- **Binding Affinity**: Use `binding_affinity` task to evaluate binding strength
- **Visualization**: Use `visualize_protein` to view the structure
- **Analysis**: Analyze interface residues and contacts

## Technical Details

### tFold Models

tFold uses two model architectures:

1. **tFold-AB**: For antibody-only prediction
   - ESM-PPI 650M for sequence encoding
   - Structure trunk for coordinate prediction

2. **tFold-Ag**: For antigen-antibody complex
   - AlphaFold2 for antigen MSA
   - ESM-PPI + tFold trunk for complex

### Sequence Format

- Input sequences should be in FASTA format (plain amino acid string)
- Use standard 20 amino acid codes
- No headers or special characters