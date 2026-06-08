---
name: antibody-design-iggm
description: >
  Antibody design using IgGM external API.
  Use this skill when:
  (1) Epitope-conditioned de novo antibody design,
  (2) Nanobody design (single chain),
  (3) Antibody affinity maturation,
  (4) Using antigen PDB structure and epitope information.

  For binding affinity evaluation, use binding-affinity-prediction-prodigy.
license: MIT
category: design-tools
tags: [structure-design, sequence-design, antibody, nanobody, iggm]
---

# IgGM Antibody De Novo Design

Design antibodies using IgGM external API for epitope-conditioned de novo design.

## API Endpoint

**IgGM API**: `http://43.142.171.112:11280/IgGM/design`

**OpenBioMed Pipeline API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/`

Environment variable override: `IGGM_API_BASE_URL` (if set, use this instead of default)

## Execution Flow

1. **Health Check**: Verify IgGM API availability
2. **Prepare Input**: Collect antigen PDB and chain sequences with X masks
3. **Construct curl Command**: Build multipart form request
4. **Execute IgGM Request**: Run curl and save JSON response
5. **Decode and Save Files**: Decode base64 PDB/FASTA content
6. **Read and Display**: Use `/run_pipeline/` to display content for remote agent

## Remote Agent Consideration

IgGM API returns base64-encoded file content in JSON response. You MUST:
1. Decode base64 content to save PDB/FASTA files
2. Use `read_protein_file` via `/run_pipeline/` to display content

**Pattern**: curl → JSON response → decode base64 → save files → `/run_pipeline/` display

## Design Types

| Type | Input | light_chain_mask | antibody_type |
|------|-------|------------------|---------------|
| Heavy-Light Antibody | H chain + L chain | Required | heavy_light |
| Nanobody | H chain only | Not provided | nanobody |

## Step 1: Health Check

```bash
curl http://43.142.171.112:11280/IgGM/health
```

## Step 2: Nanobody Design

### Input Collection
- `antigen_pdb`: Antigen PDB file path (required)
- `heavy_chain_mask`: Heavy chain sequence with X marking design regions (required)
- `epitope`: JSON list of epitope residue numbers, e.g., `[109,110,111]` (required)
- `num_samples`: Number of design samples (optional, default 1)
- `steps`: Sampling steps (optional, default 10)

### Example Execution

Input:
```
antigen_pdb: antigen.pdb
heavy_chain_mask: QVQLVESGGDLVQSGGSLKLACAVSXXXXXXXSIGWFRQAPGKEREAVSYSXXXXXXTYYVASVKGRFTISRDNAKNTAYLQMNNLKPEDTGIYYCAAXXXXXXXXXXXXXXXXXXWGQGTQVTVSS
epitope: [109,110,111,112,113,114,115,116,117]
num_samples: 1
steps: 10
```

Execute:
```bash
# 1. Call IgGM API
curl -s -X POST http://43.142.171.112:11280/IgGM/design \
  -F "antigen_pdb=@antigen.pdb" \
  -F "heavy_chain_mask=QVQLVESGGDLVQSGGSLKLACAVSXXXXXXXSIGWFRQAPGKEREAVSYSXXXXXXTYYVASVKGRFTISRDNAKNTAYLQMNNLKPEDTGIYYCAAXXXXXXXXXXXXXXXXXXWGQGTQVTVSS" \
  -F "epitope=[109,110,111,112,113,114,115,116,117]" \
  -F "num_samples=1" \
  -F "steps=10" \
  -o design_response.json

# 2. Decode and save files from response
python3 -c "
import json, base64
with open('design_response.json') as f:
    d = json.load(f)
print('job_id:', d['job_id'])
print('antibody_type:', d['antibody_type'])
for i, seq in enumerate(d['sequences']):
    print(f'--- sample {i} ---')
    print('Heavy:', seq['heavy_chain'])
    print('Antigen:', seq['antigen'][:50], '...')
# Save PDB file
open('design_result.pdb', 'wb').write(base64.b64decode(d['pdb_files'][0]['content_base64']))
# Save FASTA file
open('design_result.fasta', 'wb').write(base64.b64decode(d['fasta_files'][0]['content_base64']))
print('Files saved: design_result.pdb, design_result.fasta')
"

# 3. Read and display via run_pipeline
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "design_result.pdb", "value": "true"}'
```

## Step 3: Heavy-Light Antibody Design

### Input Collection
- `antigen_pdb`: Antigen PDB file path (required)
- `heavy_chain_mask`: Heavy chain sequence with X marking design regions (required)
- `light_chain_mask`: Light chain sequence with X marking design regions (required)
- `epitope`: JSON list of epitope residue numbers (required)
- `num_samples`: Number of design samples (optional, default 1)
- `steps`: Sampling steps (optional, default 10)

### Example Execution

Input:
```
antigen_pdb: antigen.pdb
heavy_chain_mask: VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS
light_chain_mask: DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK
epitope: [7,8,9,10,11,12,13,14,108,109,110,111,112,113,114,115,116,118,167,157,158,160,161,162,163,164]
num_samples: 1
steps: 10
```

Execute:
```bash
# 1. Call IgGM API
curl -s -X POST http://43.142.171.112:11280/IgGM/design \
  -F "antigen_pdb=@antigen.pdb" \
  -F "heavy_chain_mask=VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS" \
  -F "light_chain_mask=DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK" \
  -F "epitope=[7,8,9,10,11,12,13,14,108,109,110,111,112,113,114,115,116,118,167,157,158,160,161,162,163,164]" \
  -F "num_samples=1" \
  -F "steps=10" \
  -o design_response.json

# 2. Decode and save files
python3 -c "
import json, base64
with open('design_response.json') as f:
    d = json.load(f)
print('job_id:', d['job_id'])
print('antibody_type:', d['antibody_type'])
for i, seq in enumerate(d['sequences']):
    print(f'--- sample {i} ---')
    print('Heavy:', seq['heavy_chain'])
    print('Light:', seq['light_chain'])
    print('Antigen:', seq['antigen'][:50], '...')
# Save files
open('design_result.pdb', 'wb').write(base64.b64decode(d['pdb_files'][0]['content_base64']))
open('design_result.fasta', 'wb').write(base64.b64decode(d['fasta_files'][0]['content_base64']))
print('Files saved: design_result.pdb, design_result.fasta')
"

# 3. Read and display via run_pipeline
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "design_result.pdb", "value": "true"}'
```

## Step 4: Decode Base64 Files

IgGM returns base64-encoded file content. Use Python to decode and save.

### Decode Script

```python
import json, base64

# Load response
with open('design_response.json') as f:
    d = json.load(f)

# Display summary
print('job_id:', d['job_id'])
print('antibody_type:', d['antibody_type'])

# Display sequences
for i, seq in enumerate(d['sequences']):
    print(f'--- sample {i} ---')
    print('Heavy:', seq['heavy_chain'])
    if seq['light_chain']:
        print('Light:', seq['light_chain'])
    print('Antigen:', seq['antigen'][:50], '...')

# Save PDB file
for pdb_file in d['pdb_files']:
    filename = pdb_file['filename']
    content = base64.b64decode(pdb_file['content_base64'])
    with open(filename, 'wb') as f:
        f.write(content)
    print(f'Saved: {filename}')

# Save FASTA file
for fasta_file in d['fasta_files']:
    filename = fasta_file['filename']
    content = base64.b64decode(fasta_file['content_base64'])
    with open(filename, 'wb') as f:
        f.write(content)
    print(f'Saved: {filename}')
```

## Step 5: Read and Display for Remote Agent

After saving PDB file, call `/run_pipeline/` to display content.

### API Endpoint

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{"task": "read_protein_file", "protein": "<PDB_FILE_PATH>", "value": "true"}'
```

### Response Structure

```json
{
  "task": "read_protein_file",
  "sequence": "QVQLVESGGDLVQSGGSLKLACAVS...",
  "name": "design_result",
  "pdb_content": "REMARK 250\nATOM ...",
  "description": "Protein content read from design_result.pdb: sequence length=..."
}
```

## Response JSON Structure

IgGM API returns:

```json
{
  "job_id": "4d617f6c-54ca-4329-b5ef-1a17f69badb5",
  "antibody_type": "nanobody",
  "sequences": [
    {
      "heavy_chain": "QVQLVESGGDLVQSGGSLKLACAVS...",
      "light_chain": null,
      "antigen": "NLCPFDEVFDATRFASVYAWNRK..."
    }
  ],
  "pdb_files": [
    {
      "filename": "output_0.pdb",
      "content_base64": "RE1BUkUgMjUwIFN0cnVjdHVyZSBwcmVkaWN0ZWQgYnk..."
    }
  ],
  "fasta_files": [
    {
      "filename": "output_0.fasta",
      "content_base64": "PkhRDlFWUUxWRVNHR0RMVlFTR1N...="
    }
  ]
}
```

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `antigen_pdb` | File | ✓ | Antigen PDB file upload |
| `heavy_chain_mask` | string | ✓ | Heavy chain with X for design regions |
| `light_chain_mask` | string | - | Light chain (omit for nanobody) |
| `epitope` | JSON list | ✓ | Epitope residue numbers, e.g., `[109,110,111]` |
| `num_samples` | int | - | Number of samples (default 1) |
| `steps` | int | - | Sampling steps (default 10) |
| `task` | string | - | design/inverse_design/fr_design (default design) |
| `temperature` | float | - | Sampling temperature (default 1.0) |
| `relax` | bool | - | Structure relaxation (default false) |
| `antigen_chain_id` | string | - | Antigen chain ID in PDB (default "A") |

## Sequence Mask Format

- Use `X` to mark regions to be designed (typically CDR regions)
- Other amino acids remain fixed
- Example: `QVQLVESGGDLVQSGGSLKLACAVSXXXXXXXSIGWFRQAPGK...`

## Epitope Format

- JSON array format: `[109,110,111,112,113]`
- Must be valid residue numbers in the antigen PDB

## Error Handling

### File Upload Error

**Symptom**: API returns error about missing file

**Solution**: Ensure `antigen_pdb` file exists and use `@` prefix for upload:
```bash
-F "antigen_pdb=@/path/to/file.pdb"
```

### Epitope Format Error

**Symptom**: API returns invalid epitope error

**Solution**: Use JSON array format with square brackets:
```bash
-F "epitope=[109,110,111]"
```

### GPU Timeout

**Symptom**: Request takes very long

**Solution**: Reduce `num_samples` or `steps`, wait and retry

## Decision Tree

```
What design type?
│
├─ Nanobody (single chain)
│   └─ Only provide heavy_chain_mask
│   └─ antibody_type = nanobody
│
├─ Heavy-Light Antibody
│   └─ Provide both heavy_chain_mask and light_chain_mask
│   └─ antibody_type = heavy_light
│
└─ Need multiple samples?
    └─ Increase num_samples parameter
```

## Next Steps

After antibody design:
1. **Binding Affinity**: Use `binding-affinity-prediction-prodigy` to evaluate
2. **Visualization**: Open PDB in PyMol
3. **Structure Validation**: Use `antibody-structure-prediction-tfold` for comparison