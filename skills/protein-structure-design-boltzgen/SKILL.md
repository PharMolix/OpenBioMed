---
name: protein-structure-design-boltzgen
description: >
  All-atom protein design using BoltzGen diffusion model.
  Use this skill when:
  (1) Need side-chain aware design from the start,
  (2) Designing around small molecules or ligands,
  (3) Want all-atom diffusion (not just backbone),
  (4) Require precise binding geometries,
  (5) Using YAML-based configuration.

  For structure validation, use boltz-2.
license: MIT
category: design-tools
tags: [structure-design, sequence-design, diffusion, all-atom, binder]
---

# BoltzGen All-Atom Design

All-atom protein/peptide design via BoltzGen diffusion model, powered by the OpenBioMed `/run_pipeline/` API.

## API Endpoints

**OpenBioMed Run Pipeline API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/`

**OpenBioMed Upload API**: `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload`

## When to Use

- Design a protein binder for a target protein or peptide
- Design a protein that binds a small molecule
- Design cyclic peptides or stapled peptides
- Design nanobody/antibody CDRs
- Need all-atom precision (not just backbone)

## BoltzGen Service Limits

| Field | Value |
|-------|-------|
| Max concurrent jobs | 1 |
| Max queue size | 5 |
| Timeout | Returns `503` if queue is full |
| Typical runtime | 12-45 minutes |

## Workflow

### Step 1: Prepare Design YAML

Write a BoltzGen design specification YAML file based on the protocol and entities below.

**YAML file path conventions:**
- File paths in YAML (`path:` field) are relative to the YAML file location
- Residue indices use `label_seq_id` (1-indexed), not `auth_seq_id`

#### Entity Types

| Entity | YAML Key | Description |
|--------|----------|-------------|
| Designed protein | `protein` | Variable-length binder (`sequence: 80..140`) |
| Fixed protein | `protein` | Known sequence (`sequence: AAVTTTTPPP`) |
| Target from file | `file` | CIF/PDB file with `include`, `binding_types` |
| Small molecule | `ligand` | By SMILES (`smiles: "CCO"`) or CCD code (`ccd: ATP`) |

#### Sequence Specification

| Format | Meaning | Example |
|--------|---------|---------|
| `80..140` | Random length between 80-140 residues | `sequence: 80..140` |
| `80` | Exactly 80 designed residues | `sequence: 80` |
| `AAAVVV20PPP` | Specific residues with designed in middle | `sequence: AAAVVV20PPP` |
| `3..5C6C3` | Designed residues with specific cysteines | `sequence: 3..5C6C3` |

#### Design Protocols

| Protocol | Use Case |
|----------|----------|
| `protein-anything` | Design proteins to bind proteins or peptides |
| `peptide-anything` | Design cyclic peptides to bind proteins |
| `protein-small_molecule` | Design proteins to bind small molecules |
| `nanobody-anything` | Design nanobody CDRs |
| `antibody-anything` | Design antibody CDRs |

#### Example YAML — Protein Binder (requires CIF target file)

```yaml
entities:
  - protein:
      id: B
      sequence: 80..120
  - file:
      path: target.cif
      include:
        - chain:
            id: A
      binding_types:
        - chain:
            id: A
            binding: 45,67,89
```

#### Example YAML — Small Molecule Binding (no CIF file needed)

```yaml
entities:
  - protein:
      id: A
      sequence: 100..150
  - ligand:
      smiles: "c1ccccc1"
      id: L
```

#### Constraints (Optional)

```yaml
constraints:
  - bond:
      atom1: [S, 11, SG]    # [chain_id, res_index, atom_name]
      atom2: [S, 18, SG]    # Disulfide bond
```

#### Advanced Options

| Option | YAML Key | Description |
|--------|----------|-------------|
| Partial flexibility | `structure_groups` | `visibility: 1` = fixed, `visibility: 0` = flexible |
| Redesign residues | `design` | Specify residues to redesign on target |
| Secondary structure | `secondary_structure` | `helix`, `sheet`, `loop` constraints |
| Not-binding regions | `not_binding` | `"all"` to exclude binding to a chain |

### Step 2: Upload Files

Upload the YAML file and any CIF/PDB target files to the OpenBioMed server.

When the user has uploaded a file, you will see a file_id (UUID format) in the conversation. Use the `http_request` tool with the `files` parameter to upload it to the OpenBioMed server:

**Upload design YAML file:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<yaml_file_id>"}'
```

Response: `{"path": "./tmp/uploads/<uuid>.yaml"}`

**Upload target CIF/PDB file:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<cif_file_id>"}'
```

Response: `{"path": "./tmp/uploads/<uuid>.cif"}`

The system will automatically:
- Resolve the file_id to the actual file on disk
- Read the file bytes and send as multipart/form-data
- Inject the required API Key header

### Step 3: Submit Design Job via run_pipeline

Use the `/run_pipeline/` endpoint to submit the BoltzGen design job. The pipeline will:
1. Upload files to BoltzGen service
2. Submit the design job
3. Poll status automatically (12-45 min wait)
4. Download all result files
5. Return the final design and metrics

**API call format:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_structure_design",
  "boltzgen_yaml_file": "<yaml_server_path>",
  "boltzgen_protocol": "<protocol>",
  "boltzgen_num_designs": <num_designs>,
  "boltzgen_budget": <budget>,
  "boltzgen_cif_files": ["<cif_server_path_1>", "<cif_server_path_2>"],  // optional
  "boltzgen_output_name": "<output_name>"  // optional
}
```

#### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `boltzgen_yaml_file` | Yes | — | Design YAML file path (server path from upload) |
| `boltzgen_protocol` | No | `protein-anything` | Design protocol (see table above) |
| `boltzgen_num_designs` | No | `10` | Number of intermediate designs (production: 10000-60000) |
| `boltzgen_budget` | No | `2` | Final diversity-optimized set size |
| `boltzgen_cif_files` | No | — | List of CIF/PDB target file paths |
| `boltzgen_output_name` | No | auto-generated | Output file name prefix |

#### Response

```json
{
  "task": "boltzgen_structure_design",
  "protocol": "protein-anything",
  "output_files": [
    "./tmp/boltzgen/<output_name>/output/design.cif",
    "./tmp/boltzgen/<output_name>/output/intermediate_designs_inverse_folded/aggregate_metrics_analyze.csv",
    ...
  ],
  "description": "BoltzGen Protein-anything design completed.\nJob ID: 54c8cd3f6cca\nDesign structure: ./tmp/boltzgen/.../design.cif\n..."
}
```

**Note:** The pipeline automatically handles the 12-45 minute wait time. The response will be returned once the design completes.

### Step 4: Interpret Results

#### Output Directory Structure

```
output/
├── design.cif                # Final best design (all-atom CIF)
├── status.json               # Pipeline status
├── steps.yaml                # Steps configuration
├── intermediate_designs/     # Raw diffusion outputs
│   ├── design_0.cif
│   └── ...
├── intermediate_designs_inverse_folded/
│   ├── design_0.cif          # Refolded complexes
│   ├── aggregate_metrics_analyze.csv
│   └── ...
```

#### Quality Metrics

| Metric | Good Threshold | Interpretation |
|--------|---------------|----------------|
| Refolding RMSD | < 2.0 Å | Design folds as predicted |
| ipTM | > 0.5 | Confident interface |
| pAE | < 10 | Low alignment error |

## Example Usage

### Example 1: Small Molecule Binding (benzene)

**Input:** Design a 100-150 residue protein that binds benzene. User uploads a design YAML file.

**Step 1 — Upload YAML to OpenBioMed server:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<yaml_file_id>"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.yaml"}`

**Step 2 — Submit to run_pipeline:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_structure_design",
  "boltzgen_yaml_file": "./tmp/uploads/<uuid>.yaml",
  "boltzgen_protocol": "protein-small_molecule",
  "boltzgen_num_designs": 10,
  "boltzgen_budget": 2
}
```
→ Response: `{"task": "boltzgen_structure_design", "output_files": [...], "description": "..."}`

> The pipeline will automatically wait for the design to complete (12-45 minutes).

### Example 2: Protein Binder

**Input:** Design an 80-120 residue binder for chain A of a target protein. User uploads YAML and CIF files.

**Step 1 — Upload YAML and CIF to OpenBioMed server:**

Upload YAML:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<yaml_file_id>"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.yaml"}`

Upload CIF:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<cif_file_id>"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.cif"}`

**Step 2 — Submit to run_pipeline:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_structure_design",
  "boltzgen_yaml_file": "./tmp/uploads/<uuid>.yaml",
  "boltzgen_protocol": "protein-anything",
  "boltzgen_num_designs": 10,
  "boltzgen_budget": 2,
  "boltzgen_cif_files": ["./tmp/uploads/<uuid>.cif"]
}
```
→ Response: `{"task": "boltzgen_structure_design", "output_files": [...], "description": "..."}`

### Example 3: Cyclic Peptide with Disulfide

**Input:** Design a 10-14 residue cyclic peptide with cysteine constraints. User uploads YAML and CIF.

**Step 1 — Upload files:**

Upload YAML:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<yaml_file_id>"}'
```

Upload CIF:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<cif_file_id>"}'
```

**Step 2 — Submit:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_structure_design",
  "boltzgen_yaml_file": "./tmp/uploads/<uuid>.yaml",
  "boltzgen_protocol": "peptide-anything",
  "boltzgen_num_designs": 10,
  "boltzgen_budget": 2,
  "boltzgen_cif_files": ["./tmp/uploads/<uuid>.cif"]
}
```

## Expected Outputs

| Output | Type | Description |
|--------|------|-------------|
| `design.cif` | CIF file | Final best all-atom design structure |
| `intermediate_designs/*.cif` | CIF files | Raw diffusion outputs (num_designs count) |
| `intermediate_designs_inverse_folded/*.cif` | CIF files | Refolded complex structures |
| `aggregate_metrics_analyze.csv` | CSV | Quality metrics for all designs |
| `status.json` | JSON | Pipeline completion status |
| `steps.yaml` | YAML | Steps configuration used |

**CIF to PDB conversion** (if needed):
```python
from Bio.PDB import MMCIFParser, PDBIO
parser = MMCIFParser()
structure = parser.get_structure("design", "design.cif")
io = PDBIO()
io.set_structure(structure)
io.save("design.pdb")
```

## Error Handling

| Symptom | Cause | Solution |
|---------|-------|----------|
| Upload returns 4xx/5xx | Upload API error | Retry the upload |
| `503 Service Unavailable` | BoltzGen queue full (5 max) | Wait for existing jobs to finish |
| Pipeline timeout | Design took > 1 hour | Try simpler design or fewer num_designs |
| `CUDA out of memory` | Large design or long protein | Use fewer `num_designs` or shorter sequence range |
| `FileNotFoundError: *.cif` | Target file not provided | Ensure CIF file path is correct |
| `ValueError: invalid chain` | Chain not in target | Verify chain IDs in CIF file |
| Wrong binding site | Wrong residue indices | Use `label_seq_id` (1-indexed), verify in Molstar |

## Decision Tree

```
Should I use BoltzGen?
│
└─ What type of design?
   ├─ All-atom precision needed → protein-structure-design-boltzgen ✓
   ├─ Ligand binding pocket → protein-structure-design-boltzgen ✓
   ├─ Antibody or nanobody CDR → antibody-design-iggm
   └─ Just backbone validation → structure-prediction-boltz-2
```

## Next Steps

Validate designs with **structure-prediction-boltz-2** skill to confirm predicted structures fold correctly.