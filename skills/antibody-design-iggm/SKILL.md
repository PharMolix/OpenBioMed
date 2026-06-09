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

Design antibodies using IgGM external API for epitope-conditioned de novo design via the `/run_pipeline/` endpoint.

## API Endpoints

**OpenBioMed Pipeline API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/`

Environment variable override:
- `IGGM_API_BASE_URL`: Override IgGM API base URL
- `PIPELINE_API_URL`: Override OpenBioMed pipeline API URL

## Execution Flow

1. **Health Check**: Verify IgGM API availability (optional)
2. **Prepare Input**: Collect antigen PDB, chain masks, and epitope information
3. **Execute Design**: Call `/run_pipeline/` with `task: iggm_antibody_design`
4. **Decode Files**: IgGM returns base64-encoded content, decoded to PDB/FASTA files
5. **Save and Display**: Files saved to `./tmp/iggm/`, read for display

## Design Types

| Type | Required Inputs | Output |
|------|-----------------|--------|
| nanobody | antigen_pdb, heavy_chain_mask, epitope | PDB + FASTA |
| heavy_light | antigen_pdb, heavy_chain_mask, light_chain_mask, epitope | PDB + FASTA |

## Step 1: Nanobody Design

### Input Collection
- `task`: `"iggm_antibody_design"` (OpenBioMed task name)
- `design_type`: `"nanobody"`
- `antigen_pdb`: Antigen PDB file path (required)
- `heavy_chain_mask`: Heavy chain sequence with X marking design regions (required)
- `epitope`: JSON list of epitope residue numbers, e.g., `[109,110,111]` (required)
- `num_samples`: Number of design samples (default: 1)
- `steps`: Sampling steps (default: 10)
- `antigen_chain_id`: Antigen chain ID in PDB (default: `"A"`)
- `output_name`: Output file name prefix (optional)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "iggm_antibody_design",
    "design_type": "nanobody",
    "antigen_pdb": "./tmp/antigen.pdb",
    "heavy_chain_mask": "QVQLVESGGDLVQSGGSLKLSCAVSXXXXXXXSIGWFRQAPGKEREAVSYSXXXXXXTYYVASVKGRFTISRDNAKNTAYLQMNNLKPEDTGIYYCAAXXXXXXXXXXXXXXXXXXWGQGTQVTVSS",
    "epitope": "[109,110,111,112,113,114,115,116,117]",
    "num_samples": 1,
    "steps": 10,
    "output_name": "my_nanobody"
  }'
```

### Response

```json
{
  "task": "iggm_antibody_design",
  "design_type": "nanobody",
  "output_files": [
    "./tmp/iggm/my_nanobody_0.pdb",
    "./tmp/iggm/my_nanobody_0.fasta",
    "./tmp/iggm/my_nanobody_result.json"
  ],
  "description": "Nanobody antibody design completed.\nJob ID: 4d617f6c-54ca...\nHeavy chain: QVQLVESGGDLVQSGGSLKLSCAVS..."
}
```

## Step 2: Heavy-Light Antibody Design

### Input Collection
- `task`: `"iggm_antibody_design"` (OpenBioMed task name)
- `design_type`: `"heavy_light"`
- `antigen_pdb`: Antigen PDB file path (required)
- `heavy_chain_mask`: Heavy chain sequence with X marks (required)
- `light_chain_mask`: Light chain sequence with X marks (required)
- `epitope`: JSON list of epitope residue numbers (required)
- `num_samples`: Number of design samples (default: 1)
- `steps`: Sampling steps (default: 10)
- `output_name`: Output file name prefix (optional)

### API Call

```bash
curl -X POST http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/ \
  -H "Content-Type: application/json" \
  -d '{
    "task": "iggm_antibody_design",
    "design_type": "heavy_light",
    "antigen_pdb": "./tmp/antigen.pdb",
    "heavy_chain_mask": "VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS",
    "light_chain_mask": "DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK",
    "epitope": "[7,8,9,10,11,12,13,14,108,109,110,111,112,113,114,115,116]",
    "num_samples": 1,
    "steps": 10,
    "output_name": "my_antibody"
  }'
```

### Response

```json
{
  "task": "iggm_antibody_design",
  "design_type": "heavy_light",
  "output_files": [
    "./tmp/iggm/my_antibody_0.pdb",
    "./tmp/iggm/my_antibody_0.fasta",
    "./tmp/iggm/my_antibody_result.json"
  ],
  "description": "Heavy-light antibody design completed.\nJob ID: ...\nHeavy chain: VQLVESGG...\nLight chain: DIQMTQ..."
}
```

## Sequence Mask Format

Use `X` to mark regions to be designed (typically CDR regions):
- Other amino acids remain fixed (affinity maturation scenario)
- All positions can be marked with X for full de novo design

Example with CDR3 marked:
```
QVQLVESGGDLVQSGGSLKLSCAVSGFTFSSYAMSWVRQAPGKGLEWVAISSSGGSTYYADSVKGRLTISRDNAKNTVYLQMNSLKPEDTAVYYCAAVSYLSTASSLDYXXXXXXXXXXWGQGTQVTVSS
```

## Epitope Format

- JSON array: `[109,110,111,112,113]`
- Comma-separated string: `109,110,111,112,113`
- Must be valid residue numbers in the antigen PDB

## Output Interpretation

### Generated Files

| File Type | Content |
|-----------|---------|
| `.pdb` | Designed antibody 3D structure |
| `.fasta` | Designed sequences in FASTA format |
| `_result.json` | Job metadata and all sequences |

### JSON Result Structure

```json
{
  "job_id": "4d617f6c-54ca-4329-b5ef-1a17f69badb5",
  "antibody_type": "nanobody",
  "sequences": [
    {
      "heavy_chain": "QVQLVESGGDLVQSGGSLKLSCAVS...",
      "light_chain": null,
      "antigen": "NLCPFDEVFDATRFASVYAWNRK..."
    }
  ],
  "pdb_files": ["output_0.pdb"],
  "fasta_files": ["output_0.fasta"]
}
```

## Error Handling

### Missing Required Parameters

**Symptom**: Error message about missing inputs

**Solution**: Ensure all required parameters are provided:
- nanobody: antigen_pdb + heavy_chain_mask + epitope
- heavy_light: antigen_pdb + heavy_chain_mask + light_chain_mask + epitope

### File Not Found

**Symptom**: FileNotFoundError for antigen PDB

**Solution**: Verify antigen_pdb path exists and is accessible

### Epitope Format Error

**Symptom**: Invalid epitope format error

**Solution**: Use JSON array `[109,110,111]` or comma-separated `109,110,111`

### GPU Timeout

**Symptom**: Request takes very long

**Solution**: Reduce num_samples or steps, wait and retry

## Decision Tree

```
What design type?
│
├─ Nanobody (single chain)
│   └─ Only provide heavy_chain_mask
│   └─ design_type = "nanobody"
│
├─ Heavy-Light Antibody
│   └─ Provide both heavy_chain_mask and light_chain_mask
│   └─ design_type = "heavy_light"
│
└─ Need multiple samples?
    └─ Increase num_samples parameter
```

## Next Steps

After antibody design:
1. **Binding Affinity**: Use `binding-affinity-prediction-prodigy` to evaluate
2. **Structure Prediction**: Use `antibody-structure-prediction-tfold` for comparison
3. **Visualization**: Open PDB in PyMol to examine design