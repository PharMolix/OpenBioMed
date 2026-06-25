---
name: protein-structure-design-boltzgen
description: >
  All-atom protein design using BoltzGen diffusion model via API.
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

All-atom protein/peptide design via BoltzGen diffusion model, submitted as async jobs through the BoltzGen service API.

## API Endpoints

**BoltzGen Service API**: `http://172.16.20.44:10002`

Environment variable override:
- `BOLTZGEN_API_BASE_URL`: Override BoltzGen service base URL (default: `http://172.16.20.44:10002`)

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

### Step 2: Upload Files and Submit Job

Submit the YAML and any referenced CIF/PDB files to the BoltzGen service.

#### Input File Handling

| Input Type | How to Handle |
|------------|---------------|
| **Uploaded file (file_id)** | Use `http_request` with `files` parameter to upload to OpenBioMed server, then use server path |
| Local YAML file | Write locally, upload to OpenBioMed server via `http_request` with `files` parameter |
| Local CIF/PDB file | Upload to OpenBioMed server via `http_request` with `files` parameter |

#### Uploading User Files

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

#### Submit to BoltzGen Service

After uploading, use the returned server paths to submit the design job via `http_request`:

**Without CIF files (small molecule protocol):**

```
url: "http://172.16.20.44:10002/jobs"
method: "POST"
files: '{"design_yaml": "<yaml_file_id>"}'
body: '{"protocol": "protein-small_molecule", "num_designs": "10", "budget": "2"}'
```

**With CIF target files (protein/peptide binder protocol):**

```
url: "http://172.16.20.44:10002/jobs"
method: "POST"
files: '{"design_yaml": "<yaml_file_id>", "files": "<cif_file_id>"}'
body: '{"protocol": "protein-anything", "num_designs": "10", "budget": "2"}'
```

**With multiple CIF files:**

```
url: "http://172.16.20.44:10002/jobs"
method: "POST"
files: '{"design_yaml": "<yaml_file_id>", "files": "<cif_file_id_1>", "files": "<cif_file_id_2>"}'
body: '{"protocol": "protein-anything", "num_designs": "10", "budget": "2"}'
```

#### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `design_yaml` | Yes | — | BoltzGen design YAML file (multipart file field) |
| `files` | No | — | Referenced CIF/PDB files (multipart file field, can upload multiple) |
| `protocol` | No | `protein-anything` | Design protocol (see table above) |
| `num_designs` | No | `10` | Number of intermediate designs (production: 10000-60000) |
| `budget` | No | `2` | Final diversity-optimized set size |
| `extra_args` | No | — | Extra CLI arguments, space-separated |

#### Response

```json
{
  "job_id": "54c8cd3f6cca",
  "status": "pending",
  "output_dir": "/home/sulixian/boltzGen/runs/54c8cd3f6cca/output",
  "input_dir": "/home/sulixian/boltzGen/runs/54c8cd3f6cca/input",
  "log_file": "/home/sulixian/boltzGen/runs/54c8cd3f6cca/log.txt"
}
```

### Step 3: Check Job Status

BoltzGen jobs are **async** — they run for 12-45 minutes depending on design complexity. Poll status until `succeeded` or `failed`.

```
url: "http://172.16.20.44:10002/jobs/{job_id}"
method: "GET"
```

#### Status Values

| Status | Meaning | Action |
|--------|---------|--------|
| `pending` | Queued, not yet started | Wait and re-check later |
| `running` | Currently executing | Wait and re-check later (typical: 12-45 min) |
| `succeeded` | Completed, `return_code: 0` | Proceed to Step 4 |
| `failed` | Error occurred | Check `error_message` field and logs |

**When the job is `running`:** Remind the user that BoltzGen design takes 12-45 minutes. Suggest they check back later using:
```
url: "http://172.16.20.44:10002/jobs/{job_id}"
method: "GET"
```

**View progress logs:**

```
url: "http://172.16.20.44:10002/jobs/{job_id}/log"
method: "GET"
```

**Cancel a running or queued job:**

```
url: "http://172.16.20.44:10002/jobs/{job_id}/cancel"
method: "POST"
```

### Step 4: Download Results

Once `status: succeeded`, download the final design and metrics.

**List all result files:**

```
url: "http://172.16.20.44:10002/jobs/{job_id}/results"
method: "GET"
```

**Download all results as zip:**

```
url: "http://172.16.20.44:10002/jobs/{job_id}/download"
method: "GET"
```

**Download single result file (e.g., design.cif):**

```
url: "http://172.16.20.44:10002/jobs/{job_id}/files/design.cif"
method: "GET"
```

### Step 5: Interpret Results

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

### Example 1: Small Molecule Binding (benzene) — with uploaded YAML

**Input:** Design a 100-150 residue protein that binds benzene. User uploads a design YAML file.

**Step 1 — Upload YAML to OpenBioMed server:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<yaml_file_id>"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.yaml"}`

**Step 2 — Submit to BoltzGen:**

```
url: "http://172.16.20.44:10002/jobs"
method: "POST"
files: '{"design_yaml": "<yaml_file_id>"}'
body: '{"protocol": "protein-small_molecule", "num_designs": "10", "budget": "2"}'
```
→ Response: `{"job_id": "54c8cd3f6cca", "status": "pending"}`

**Step 3 — Check status (running):**

```
url: "http://172.16.20.44:10002/jobs/54c8cd3f6cca"
method: "GET"
```
→ Response: `{"job_id": "54c8cd3f6cca", "status": "running", ...}`

> BoltzGen design is running. This typically takes 12-45 minutes. Please check back later by querying:
> `url: "http://172.16.20.44:10002/jobs/54c8cd3f6cca"`, `method: "GET"`

**Step 3 — Check status (completed):**

```
url: "http://172.16.20.44:10002/jobs/54c8cd3f6cca"
method: "GET"
```
→ Response: `{"job_id": "54c8cd3f6cca", "status": "succeeded", "return_code": 0}`

**Step 4 — Download design.cif:**

```
url: "http://172.16.20.44:10002/jobs/54c8cd3f6cca/files/design.cif"
method: "GET"
```

### Example 2: Protein Binder — with uploaded YAML and CIF

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

**Step 2 — Submit to BoltzGen:**

```
url: "http://172.16.20.44:10002/jobs"
method: "POST"
files: '{"design_yaml": "<yaml_file_id>", "files": "<cif_file_id>"}'
body: '{"protocol": "protein-anything", "num_designs": "10", "budget": "2"}'
```
→ Response: `{"job_id": "abc123", "status": "pending"}`

**Step 3 — Check status:**

```
url: "http://172.16.20.44:10002/jobs/abc123"
method: "GET"
```

### Example 3: Cyclic Peptide with Disulfide — with uploaded files

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
url: "http://172.16.20.44:10002/jobs"
method: "POST"
files: '{"design_yaml": "<yaml_file_id>", "files": "<cif_file_id>"}'
body: '{"protocol": "peptide-anything", "num_designs": "10", "budget": "2"}'
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
| Upload returns 4xx/5xx | Upload API error | Retry the upload. The system handles multipart encoding and API key automatically |
| `503 Service Unavailable` on submit | Queue full (5 max) | Wait for existing jobs to finish, or cancel queued jobs |
| `status: failed` | Runtime error | Check `error_message` field and `/jobs/{id}/log` |
| `CUDA out of memory` in log | Large design or long protein | Use fewer `num_designs` or shorter sequence range |
| `FileNotFoundError: *.cif` | Target file not uploaded | Ensure CIF file is uploaded via `files` parameter |
| `ValueError: invalid chain` | Chain not in target | Verify chain IDs in CIF file |
| Wrong binding site | Wrong residue indices | Use `label_seq_id` (1-indexed), verify in Molstar |

## Admin Operations

```
url: "http://172.16.20.44:10002/health"
method: "GET"
```

```
url: "http://172.16.20.44:10002/queue"
method: "GET"
```

```
url: "http://172.16.20.44:10002/admin/unload"
method: "POST"
```

```
url: "http://172.16.20.44:10002/admin/cleanup?max_age_days=7"
method: "POST"
```

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
