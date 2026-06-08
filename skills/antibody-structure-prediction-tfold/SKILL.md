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

Predict antibody structure, antigen-antibody complex structure, and epitope residues using external tFold API via curl commands.

## API Endpoints

**tFold API**: `http://43.142.171.112:11280/tFold`

**OpenBioMed Pipeline API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/`

Environment variable override: `TFOLD_API_BASE_URL` (if set, use this instead of default)

## Execution Flow

1. **Health Check**: Verify tFold API availability
2. **Prepare Input**: Collect sequences based on task type
3. **Construct curl Command**: Build JSON payload for tFold
4. **Execute tFold Request**: Run curl and save PDB file
5. **Read and Display**: Call `/run_pipeline/` with `read_protein_file` task to display content

## Remote Agent Consideration

When running as a remote agent, the file system is not accessible. After saving the PDB file, call `/run_pipeline/` endpoint with `task: read_protein_file` to read and display the protein content.

**Pattern**: tFold curl → Save PDB → `/run_pipeline/` read_protein_file → Display

## Task Types

| Task | tFold Endpoint | Required Inputs | Output |
|------|----------------|-----------------|--------|
| antibody | `/predict/ab` | heavy_chain, light_chain | PDB file |
| nanobody | `/predict/ab` | heavy_chain (single) | PDB file |
| complex | `/predict/ag` | heavy_chain, light_chain, antigen | PDB file |
| epitope | `/predict/epitope` | pdb_file, antigen_id | JSON |

## Step 1: Health Check

Before any prediction, verify the tFold API is available:

```bash
curl http://43.142.171.112:11280/tFold/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu": "cuda:0",
  "models_loaded": ["ab", "ag"]
}
```

## Step 2: Antibody Structure Prediction

### Input Collection
- `heavy_chain`: Heavy chain FASTA sequence (required)
- `light_chain`: Light chain FASTA sequence (required)
- `output_name`: Output file name (optional, auto-generated if not provided)

### curl Command Construction

```bash
curl -X POST http://43.142.171.112:11280/tFold/predict/ab \
  -H "Content-Type: application/json" \
  -d '{"chains": [{"id": "H", "sequence": "<HEAVY_CHAIN>"}, {"id": "L", "sequence": "<LIGHT_CHAIN>"}], "output_name": "<OUTPUT_NAME>"}' \
  -o <OUTPUT_NAME>.pdb
```

### Example Execution

Input:
```
heavy_chain: EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK
light_chain: DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK
output_name: my_antibody
```

Execute:
```bash
# 1. Save PDB file from tFold
curl -X POST http://43.142.171.112:11280/tFold/predict/ab \
  -H "Content-Type: application/json" \
  -d '{"chains": [{"id": "H", "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK"}, {"id": "L", "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK"}], "output_name": "my_antibody"}' \
  -o my_antibody.pdb

# 2. Read and display via run_pipeline
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "my_antibody.pdb", "value": "true"}'
```

Response from `read_protein_file`:
```json
{
  "task": "read_protein_file",
  "sequence": "EVQLVESGGGLVQPGGSLRLSCAAS...",
  "name": "my_antibody",
  "pdb_content": "REMARK 250\nREMARK 250 Predicted lDDT-Ca score: 0.9482\n...",
  "description": "Protein content read from my_antibody.pdb: sequence length=..."
}
```

## Step 3: Nanobody Structure Prediction

### Input Collection
- `heavy_chain`: Single chain FASTA sequence (required)
- `output_name`: Output file name (optional)

### curl Command Construction

For nanobody, only provide one chain with id "H":

```bash
curl -X POST http://43.142.171.112:11280/tFold/predict/ab \
  -H "Content-Type: application/json" \
  -d '{"chains": [{"id": "H", "sequence": "<HEAVY_CHAIN>"}], "output_name": "<OUTPUT_NAME>"}' \
  -o <OUTPUT_NAME>.pdb
```

### Example Execution

Input:
```
heavy_chain: MSIQEIQKEIAQIQAVIAGIQKYIYTMSIEEIQKQIAAIQCQIAAIQKQIYAMSIEEIQKQIAAIQEQILAIYKQIMAMVT
output_name: my_nanobody
```

Execute:
```bash
# 1. Save PDB file from tFold
curl -X POST http://43.142.171.112:11280/tFold/predict/ab \
  -H "Content-Type: application/json" \
  -d '{"chains": [{"id": "H", "sequence": "MSIQEIQKEIAQIQAVIAGIQKYIYTMSIEEIQKQIAAIQCQIAAIQKQIYAMSIEEIQKQIAAIQEQILAIYKQIMAMVT"}], "output_name": "my_nanobody"}' \
  -o my_nanobody.pdb

# 2. Read and display via run_pipeline
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "my_nanobody.pdb", "value": "true"}'
```

## Step 4: Antigen-Antibody Complex Prediction

### Input Collection
- `heavy_chain`: Heavy chain FASTA sequence (required)
- `light_chain`: Light chain FASTA sequence (required)
- `antigen`: Antigen FASTA sequence (required)
- `antigen_id`: Chain ID for antigen (default: "A")
- `msa_content`: MSA content in a3m format (optional, improves accuracy)
- `output_name`: Output file name (optional)

### Mode A: Without MSA (Simpler)

```bash
curl -X POST http://43.142.171.112:11280/tFold/predict/ag \
  -H "Content-Type: application/json" \
  -d '{"antibody_chains": [{"id": "H", "sequence": "<HEAVY_CHAIN>"}, {"id": "L", "sequence": "<LIGHT_CHAIN>"}], "antigen_sequence": "<ANTIGEN>", "antigen_id": "<ANTIGEN_ID>", "output_name": "<OUTPUT_NAME>"}' \
  -o <OUTPUT_NAME>.pdb
```

### Mode B: With MSA (Recommended)

Use Python to construct JSON (handles escaping properly):

```bash
# Read MSA file
MSA_TEXT=$(cat /path/to/antigen.a3m)

# Construct JSON payload
JSON_BODY=$(python3 -c "
import json, sys
body = {
  'antibody_chains': [
    {'id': 'H', 'sequence': '<HEAVY_CHAIN>'},
    {'id': 'L', 'sequence': '<LIGHT_CHAIN>'}
  ],
  'antigen_sequence': '<ANTIGEN>',
  'antigen_id': '<ANTIGEN_ID>',
  'msa_content': sys.argv[1],
  'output_name': '<OUTPUT_NAME>'
}
print(json.dumps(body))
" "$MSA_TEXT")

# Execute curl
curl -X POST http://43.142.171.112:11280/tFold/predict/ag \
  -H "Content-Type: application/json" \
  -d "$JSON_BODY" \
  -o <OUTPUT_NAME>.pdb
```

### Example Execution (Without MSA)

Input:
```
heavy_chain: EVQLVQSGAEVKKPGESLKISCKGSGYSFSNYWIGWVRQMPGKGLEWMGIIDPSNSYTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARWYYKPFDVWGQGTLVTVSS
light_chain: QSVLTQPPSVSGAPGQRVTISCTGSSSNIGSGYDVHWYQQLPGTAPKLLIYGNSKRPSGVPDRFSGSKSGTSASLAITGLQSEDEADYYCASWTDGLSLVVFGGGTKLTVL
antigen: RAVPGGSSPAWTQCQQLSQKLCTLAWSAHPLVGHMDLREEDVPHIQCGDGCDPQGLRDNSQFCLQRIHQGLIFYEKLLGSDIFTGEPSLLPDSPVGQLHASLLGLSQLLQPEGHHWETQQIPSLSPSQPWQRLLLRFKILRSLQAFVAVAARVFAHGAATL
antigen_id: A
output_name: my_complex
```

Execute:
```bash
# 1. Save PDB file from tFold
curl -X POST http://43.142.171.112:11280/tFold/predict/ag \
  -H "Content-Type: application/json" \
  -d '{"antibody_chains': [{"id": "H", "sequence": "EVQLVQSGAEVKKPGESLKISCKGSGYSFSNYWIGWVRQMPGKGLEWMGIIDPSNSYTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARWYYKPFDVWGQGTLVTVSS"}, {"id": "L", "sequence": "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGSGYDVHWYQQLPGTAPKLLIYGNSKRPSGVPDRFSGSKSGTSASLAITGLQSEDEADYYCASWTDGLSLVVFGGGTKLTVL"}], "antigen_sequence": "RAVPGGSSPAWTQCQQLSQKLCTLAWSAHPLVGHMDLREEDVPHIQCGDGCDPQGLRDNSQFCLQRIHQGLIFYEKLLGSDIFTGEPSLLPDSPVGQLHASLLGLSQLLQPEGHHWETQQIPSLSPSQPWQRLLLRFKILRSLQAFVAVAARVFAHGAATL", "antigen_id": "A", "output_name": "my_complex"}' \
  -o my_complex.pdb

# 2. Read and display via run_pipeline
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "my_complex.pdb", "value": "true"}'
```

## Step 5: Epitope Determination

### Input Collection
- `pdb_file`: Path to antigen-antibody complex PDB file (required)
- `antigen_id`: Chain ID for antigen (default: "A")
- `distance_threshold`: Distance threshold in Angstroms (default: 5.0)

### curl Command Construction

Use multipart form upload (`-F` flag):

```bash
curl -X POST http://43.142.171.112:11280/tFold/predict/epitope \
  -F "pdb_file=@<PDB_FILE_PATH>" \
  -F "antigen_id=<ANTIGEN_ID>" \
  -F "distance_threshold=<THRESHOLD>"
```

### Example Execution

Input:
```
pdb_file: my_complex.pdb
antigen_id: A
distance_threshold: 5.0
```

Execute:
```bash
# Epitope returns JSON directly (no read_protein_file needed)
curl -X POST http://43.142.171.112:11280/tFold/predict/epitope \
  -F "pdb_file=@my_complex.pdb" \
  -F "antigen_id=A" \
  -F "distance_threshold=5.0"
```

Output (JSON displayed directly):
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

## Step 6: Read and Display for Remote Agent

After saving PDB file, call `/run_pipeline/` to read and display content.

### API Endpoint

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "<PDB_FILE_PATH>", "value": "true"}'
```

### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | string | `"read_protein_file"` |
| `protein` | string | PDB file path |
| `value` | string | `"true"` to include PDB content, `"false"` for sequence only |

### Response Structure

```json
{
  "task": "read_protein_file",
  "sequence": "EVQLVESGGGLVQPGGSLRLSCAAS...",
  "name": "my_antibody",
  "pdb_content": "REMARK 250\nREMARK 250 Predicted lDDT-Ca score: 0.9482\nATOM ...",
  "description": "Protein content read from my_antibody.pdb: sequence length=..."
}
```

### Key Information to Display

From the response:

1. **Description**: `"Protein content read from my_antibody.pdb: sequence length=..."`
2. **Sequence**: Full FASTA sequence
3. **PDB Content**: Contains REMARK lines with confidence scores

Extract confidence from `pdb_content`:
```
REMARK 250 Predicted lDDT-Ca score: 0.9482
REMARK 250 Predicted pTM score: 0.7861
REMARK 250 Predicted ipTM score: 0.8023
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

### curl Command Format Errors

**Symptom**: `-: command not found`

**Solution**: Ensure proper line continuation with `\` - no spaces after `\`

```bash
# WRONG (space after backslash):
curl -X POST URL \
  -H "Content-Type: application/json" \  # <-- space here causes error
  -d '...'

# CORRECT:
curl -X POST URL \
  -H "Content-Type: application/json" \
  -d '...'
```

### API Returns Empty Body

**Symptom**: `{"detail":[{"type":"missing","loc":["body"],"msg":"Field required"}]}`

**Solution**: Check `-d` parameter is properly formatted JSON. Use single-line or proper escaping.

### GPU Memory Timeout

**Symptom**: Request takes very long or fails

**Solution**: Wait for API to free GPU (check `/health` endpoint), then retry.

### read_protein_file Error

**Symptom**: `FileNotFoundError` in response

**Solution**: Ensure PDB file was successfully saved by tFold curl before calling read_protein_file.

## Sequence Format Requirements

- Plain amino acid string (no FASTA header)
- Standard 20 amino acid codes (ACDEFGHIKLMNPQRSTVWY)
- No spaces or special characters

## Decision Tree

```
What task type?
│
├─ Antibody structure → /predict/ab (H + L chains)
│   └─ After: read_protein_file via /run_pipeline/
│
├─ Nanobody structure → /predict/ab (single H chain)
│   └─ After: read_protein_file via /run_pipeline/
│
├─ Antigen-antibody complex → /predict/ag
│   ├─ Have MSA? → Include msa_content
│   └─ No MSA? → Single sequence mode
│   └─ After: read_protein_file via /run_pipeline/
│
└─ Epitope residues → /predict/epitope (upload PDB)
    └─ Returns JSON directly, no read step needed
```

## Next Steps

After structure prediction:
1. **Visualize**: Open PDB in PyMol or molecular viewer
2. **Binding Affinity**: Use `binding-affinity-prediction-prodigy` skill
3. **Analysis**: Examine interface residues and contacts
4. **Epitope**: If complex predicted, run epitope determination