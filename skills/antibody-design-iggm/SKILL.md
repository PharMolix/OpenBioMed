---
name: antibody-design-iggm
description: >
  Antibody design using IgGM model.
  Use this skill when:
  (1) Epitope-conditioned de novo antibody design,
  (2) Antibody affinity maturation,
  (3) Using antigen PDB structure and epitope information.

  For binding affinity evaluation, use binding-affinity-prediction-prodigy.
license: MIT
category: design-tools
tags: [structure-design, sequence-design, antibody, nanobody, iggm]
---

# IgGM Antibody De Novo Design

Design antibodies using IgGM deep learning model for epitope-conditioned de novo design and affinity maturation.

## When to Use

- User wants to design antibodies/nanobodies based on antigen epitope
- User wants affinity maturation for existing antibodies
- User provides antigen PDB structure and epitope information
- User provides design requirement FASTA (X marks design regions)

## API Endpoint Resolution

The skill resolves the OpenBioMed API base URL in this order:

1. **Environment variable**: `${OPENBIOMED_API_BASE_URL}` (if set)
2. **Docker container default**: `http://openbiomed-server:8090` (if running in Docker)
3. **Local development default**: `http://127.0.0.1:8090`

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL.

## Workflow

### Step 1: Prepare Input Files

Prepare the following input files:

| Input | Type | Description |
|-------|------|-------------|
| FASTA file | .fasta | Design requirement with X marking design regions |
| Antigen PDB | .pdb | Antigen 3D structure |
| Epitope | string | Residue numbers (space-separated, optional) |
| Original FASTA | .fasta | Original antibody for affinity maturation |

**FASTA format example**:
```
>H  # Heavy chain ID
VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS
>L # Light chain ID
DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK
>A # Antigen ID (must match PDB chain)
NLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGP
```

**Note**: 'X' indicates the region to be designed.

### Step 2: Call antibody_design API

#### De Novo Antibody Design

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "antibody_design", "fasta": "<FASTA_FILE_PATH>", "antigen_pdb": "<ANTIGEN_PDB_PATH>", "epitope": "7 8 9 10 11"}'
```

**Response**:
```json
{
  "task": "antibody_design",
  "mode": "design",
  "output_files": ["./tmp/antibody_design_xxx/antibody_0.fasta", "./tmp/antibody_design_xxx/antibody_0.pdb"],
  "description": "Antibody design completed. Output saved to ./tmp/antibody_design_xxx"
}
```

#### Antibody Affinity Maturation

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "antibody_design", "fasta": "<FASTA_FILE_PATH>", "antigen_pdb": "<ANTIGEN_PDB_PATH>", "fasta_origin": "<ORIGINAL_FASTA_PATH>", "mode": "affinity_maturation", "num_samples": 10}'
```

**Response**:
```json
{
  "task": "antibody_design",
  "mode": "affinity_maturation",
  "output_files": ["./tmp/antibody_design_xxx/antibody_0.fasta", "./tmp/antibody_design_xxx/antibody_0.pdb"],
  "description": "Antibody design completed. Output saved to ./tmp/antibody_design_xxx"
}
```

### Step 3: View and Use Results

The designed antibodies are saved as FASTA and PDB files:

```
output_dir/
├── antibody_0.fasta  # designed antibody sequence
├── antibody_0.pdb    # designed antigen-antibody complex structure
├── antibody_1.fasta
├── antibody_1.pdb
└── ...
```

You can:
1. **Visualize**: Use `visualize_complex` task or PyMol
2. **Evaluate affinity**: Use `binding_affinity` task
3. **Download**: Copy files from the server

## Example Usage

### Example 1: De Novo Antibody Design

```
Input: "Design an antibody targeting the spike protein epitope residues 7-15"

Step 1: Prepare files
  FASTA: design.fasta (with X marking CDR regions)
  Antigen PDB: spike_protein.pdb
  Epitope: 7 8 9 10 11 12 13 14 15

Step 2: Call API
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "antibody_design", "fasta": "design.fasta", "antigen_pdb": "spike_protein.pdb", "epitope": "7 8 9 10 11 12 13 14 15"}'

Output:
  Designed antibody files in ./tmp/antibody_design_xxx/
```

## Expected Outputs

| Output | Type | Description |
|--------|------|-------------|
| output_files | list | List of designed FASTA and PDB file paths |
| mode | string | "design" or "affinity_maturation" |
| description | string | Human-readable description |

## Error Handling

### Missing Input Files

**Symptom**: API returns error about missing files.

**Solution**: Ensure fasta and antigen_pdb file paths are correct and files exist.

### IgGM Not Installed

**Symptom**: API returns "IgGM not installed" error.

**Solution**: Install IgGM in the server environment:
```bash
git clone https://github.com/TencentAI4S/IgGM.git
cd IgGM
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install tqdm requests numpy==1.23.5 termcolor==2.4.0 biopython==1.79 openmm==8.2 pdbfixer ml-collections==0.1.1
```

### GPU Memory Error

**Symptom**: Design fails with CUDA out of memory.

**Solution**: IgGM requires significant GPU memory (24GB+ recommended). Try:
- Using a GPU with more memory
- Reducing num_samples parameter

### Epitope Format Error

**Symptom**: API returns error about epitope format.

**Solution**: Ensure epitope is space-separated residue numbers starting from 1 (e.g., "7 8 9 10 11").

## Decision Tree

```
Should I use IgGM?
│
└─ What type of design?
   ├─ Antibody de novo design → antibody-design-iggm ✓
   ├─ Nanobody de novo design → antibody-design-iggm ✓
   ├─ Antibody affinity maturation → antibody-design-iggm ✓
   └─ General protein binder design → boltzgen
```

## Next Steps

After antibody design:
- **Binding Affinity**: Use `binding_affinity` task to evaluate binding strength
- **Visualization**: Use `visualize_complex` to view the designed structure
- **Structure Prediction**: Use `antibody_structure` task for structure validation

## Technical Details

### IgGM Model

IgGM uses:
1. Epitope-conditioned design: Generates antibodies targeting specific epitopes
2. Structure-aware generation: Produces both sequence and 3D structure
3. Affinity maturation: Optimizes existing antibodies for better binding

### Design Modes

1. **De Novo Design**: Generate new antibodies from scratch
   - Input: FASTA with X marking CDR regions
   - Output: Multiple candidate antibodies

2. **Affinity Maturation**: Improve existing antibody binding
   - Input: FASTA with X marking optimization regions + original sequence
   - Output: Optimized antibody variants