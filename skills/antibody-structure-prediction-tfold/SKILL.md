---
name: antibody-structure-prediction-tfold
description: >
  Antibody-related structure prediction using tFold external API.
  Use this skill when:
  (1) Predict antibody and nanobody structure of a given sequence,
  (2) Predict antigen-antibody complex structure of given sequences,
  (3) Determine epitope residues from antigen-antibody complex PDB,
  (4) External GPU resources (no local model weights required).

  For binding affinity evaluation, use binding-affinity-prediction-prodigy.
license: MIT
category: design-tools
tags: [structure-prediction, antibody, nanobody, antigen-antibody complex, epitope, tfold]
---

# tFold Antibody-related Structure Prediction

Predict antibody structure, antigen-antibody complex structure, and epitope residues using external tFold API via the `/run_pipeline/` endpoint.

## API Endpoints

**OpenBioMed Pipeline API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/`

Environment variable override:
- `TFOLD_API_BASE_URL`: Override tFold API base URL
- `PIPELINE_API_URL`: Override OpenBioMed pipeline API URL

## Execution Flow

1. **Health Check**: Verify tFold API availability (optional)
2. **Prepare Input**: Collect sequences based on task type
3. **Execute Prediction**: Call `/run_pipeline/` with `task: tfold_antibody_structure`
4. **Save and Display**: PDB file saved, confidence scores extracted

## Task Types

| Task | Required Inputs | Output |
|------|-----------------|--------|
| antibody | heavy_chain, light_chain | PDB file + confidence |
| nanobody | heavy_chain (single) | PDB file + confidence |
| complex | heavy_chain, light_chain, antigen | PDB file + confidence |
| epitope | pdb_file, antigen_id | JSON with epitope residues |

## Step 1: Antibody Structure Prediction

### Input Collection
- `task`: `"tfold_antibody_structure"` (OpenBioMed task name)
- `prediction_type`: `"antibody"` (prediction mode)
- `heavy_chain`: Heavy chain FASTA sequence (required)
- `light_chain`: Light chain FASTA sequence (required)
- `output_name`: Output file name (optional, auto-generated if not provided)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "tfold_antibody_structure",
    "prediction_type": "antibody",
    "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK",
    "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
    "output_name": "my_antibody"
  }'
```

### Response

```json
{
  "results": ["./tmp/tfold/my_antibody.pdb"],
  "messages": [
    "Antibody structure prediction completed.\nPDB file: ./tmp/tfold/my_antibody.pdb\nlDDT-Ca score: 0.9482\npTM score: 0.7861\nSequence length: 224"
  ]
}
```

## Step 2: Nanobody Structure Prediction

### Input Collection
- `task`: `"tfold_antibody_structure"` (OpenBioMed task name)
- `prediction_type`: `"nanobody"` (prediction mode)
- `heavy_chain`: Single chain FASTA sequence (required)
- `output_name`: Output file name (optional)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "tfold_antibody_structure",
    "prediction_type": "nanobody",
    "heavy_chain": "MSIQEIQKEIAQIQAVIAGIQKYIYTMSIEEIQKQIAAIQCQIAAIQKQIYAMSIEEIQKQIAAIQEQILAIYKQIMAMVT",
    "output_name": "my_nanobody"
  }'
```

### Response

```json
{
  "results": ["./tmp/tfold/my_nanobody.pdb"],
  "messages": [
    "Nanobody structure prediction completed.\nPDB file: ./tmp/tfold/my_nanobody.pdb\nlDDT-Ca score: 0.8523\nSequence length: 78"
  ]
}
```

## Step 3: Antigen-Antibody Complex Prediction

### Input Collection
- `task`: `"tfold_antibody_structure"` (OpenBioMed task name)
- `prediction_type`: `"complex"` (prediction mode)
- `heavy_chain`: Heavy chain FASTA sequence (required)
- `light_chain`: Light chain FASTA sequence (required)
- `antigen`: Antigen FASTA sequence (required)
- `antigen_id`: Chain ID for antigen (default: `"A"`)
- `msa_content`: MSA content in a3m format (optional, improves accuracy)
- `output_name`: Output file name (optional)

### API Call (Without MSA)

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "tfold_antibody_structure",
    "prediction_type": "complex",
    "heavy_chain": "EVQLVQSGAEVKKPGESLKISCKGSGYSFSNYWIGWVRQMPGKGLEWMGIIDPSNSYTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARWYYKPFDVWGQGTLVTVSS",
    "light_chain": "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGSGYDVHWYQQLPGTAPKLLIYGNSKRPSGVPDRFSGSKSGTSASLAITGLQSEDEADYYCASWTDGLSLVVFGGGTKLTVL",
    "antigen": "RAVPGGSSPAWTQCQQLSQKLCTLAWSAHPLVGHMDLREEDVPHIQCGDGCDPQGLRDNSQFCLQRIHQGLIFYEKLLGSDIFTGEPSLLPDSPVGQLHASLLGLSQLLQPEGHHWETQQIPSLSPSQPWQRLLLRFKILRSLQAFVAVAARVFAHGAATL",
    "antigen_id": "A",
    "output_name": "my_complex"
  }'
```

### Response

```json
{
  "results": ["./tmp/tfold/my_complex.pdb"],
  "messages": [
    "Antigen-antibody complex structure prediction completed.\nPDB file: ./tmp/tfold/my_complex.pdb\nAntigen chain ID: A\nlDDT-Ca score: 0.9234\nipTM score: 0.8023\nTotal sequence length: 450"
  ]
}
```

### API Call (With MSA)

```bash
# First read MSA file content, then pass as parameter
MSA_CONTENT=$(cat /path/to/antigen.a3m)

curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "tfold_antibody_structure",
    "prediction_type": "complex",
    "heavy_chain": "...",
    "light_chain": "...",
    "antigen": "...",
    "antigen_id": "A",
    "msa_content": "'"$MSA_CONTENT"'",
    "output_name": "my_complex_with_msa"
  }'
```

## Step 4: Epitope Determination

### Input Collection
- `task`: `"tfold_antibody_structure"` (OpenBioMed task name)
- `prediction_type`: `"epitope"` (prediction mode)
- `pdb_file`: Path to antigen-antibody complex PDB file (required)
- `antigen_id`: Chain ID for antigen (default: `"A"`)
- `distance_threshold`: Distance threshold in Angstroms (default: `5.0`)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "tfold_antibody_structure",
    "prediction_type": "epitope",
    "pdb_file": "./tmp/tfold/my_complex.pdb",
    "antigen_id": "A",
    "distance_threshold": 5.0
  }'
```

### Response

```json
{
  "results": ["./tmp/tfold/epitope_result_1234567890.json"],
  "messages": [
    "Epitope determination completed.\nAntigen chain ID: A\nDistance threshold: 5.0 Angstroms\nEpitope residue count: 65\nEpitope residues (first 10): [27, SER, A], [28, ALA, A], ...\n... and 55 more residues"
  ]
}
```

The epitope JSON file contains:
```json
{
  "status": "success",
  "task": "Epitope",
  "antigen_id": "A",
  "distance_threshold": 5.0,
  "epitope_residues": [
    [27, "SER", "A"],
    [28, "ALA", "A"],
    ...
  ],
  "epitope_count": 65
}
```

## Output Interpretation

### PDB File Confidence Scores

Key scores in REMARK lines:
- **lDDT-Ca score** (0-1): Local structure confidence, >0.8 is high
- **pTM score** (0-1): Overall topology confidence
- **ipTM score** (0-1): Interface confidence (complex mode)

### Epitope JSON

- `epitope_residues`: `[residue_number, residue_name, chain_id]`
- `epitope_count`: Total number of epitope residues

## Error Handling

### API Unavailable

**Symptom**: Health check fails, prediction errors

**Solution**: Check network connectivity, verify tFold API URL, or wait for service restoration

### Missing Required Parameters

**Symptom**: Error message about missing required inputs

**Solution**: Ensure all required parameters for the task type are provided:
- antibody: heavy_chain + light_chain
- nanobody: heavy_chain
- complex: heavy_chain + light_chain + antigen
- epitope: pdb_file

### GPU Memory Timeout

**Symptom**: Request takes very long or fails

**Solution**: Reduce complexity, wait for API to free GPU, then retry

## Sequence Format Requirements

- Plain amino acid string (no FASTA header)
- Standard 20 amino acid codes (ACDEFGHIKLMNPQRSTVWY)
- No spaces or special characters

## Decision Tree

```
What task type?
│
├─ Antibody structure → task: "antibody" (H + L chains)
│
├─ Nanobody structure → task: "nanobody" (single H chain)
│
├─ Antigen-antibody complex → task: "complex"
│   ├─ Have MSA? → Include msa_content parameter
│   └─ No MSA? → Use single sequence mode
│
└─ Epitope residues → task: "epitope" (requires PDB file)
```

## Next Steps

After structure prediction:
1. **Visualize**: Open PDB in PyMol or molecular viewer
2. **Binding Affinity**: Use `binding-affinity-prediction-prodigy` skill
3. **Analysis**: Examine interface residues and contacts
4. **Epitope**: If complex predicted, run epitope determination