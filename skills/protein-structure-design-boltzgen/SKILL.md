---
name: protein-structure-design-boltzgen
description: >
  All-atom protein design using BoltzGen diffusion model.
  Use this skill when:
  (1) Need side-chain aware design from the start,
  (2) Designing around small molecules or ligands,
  (3) Want all-atom diffusion (not just backbone),
  (4) Require precise binding geometries.

  For structure validation, use boltz-2.
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

### Step 1: Conversational Configuration → Generate YAML

**DO NOT ask user to upload YAML file.** Instead, guide the conversation to collect design parameters and generate the YAML configuration.

#### Conversational Guide

**Reference**: `references/parameters_guide.md` → Section 1: Conversational Configuration Guide

Ask questions based on the user's design target:

| Target Type | Protocol | Questions to Ask |
|-------------|----------|------------------|
| Protein/Peptide | `protein-anything` | Target CIF path, chain ID, binding residues, binder chain ID, binder length |
| Cyclic Peptide | `peptide-anything` | Target CIF path, chain ID, binding residues, peptide length, disulfide positions |
| Small Molecule | `protein-small_molecule` | SMILES or CCD code, ligand ID, protein ID, protein length |
| Nanobody | `nanobody-anything` | Framework sequence, chain ID, antigen CIF, antigen chain |
| Antibody | `antibody-anything` | Heavy chain sequence, light chain sequence, antigen CIF, epitope residues |

#### Quick Protocol Selection

| Protocol | Use Case |
|----------|----------|
| `protein-anything` | Design proteins to bind proteins or peptides |
| `peptide-anything` | Design cyclic peptides to bind proteins |
| `protein-small_molecule` | Design proteins to bind small molecules |
| `nanobody-anything` | Design nanobody CDRs |
| `antibody-anything` | Design antibody CDRs |

#### YAML Templates

**Reference**: `references/yaml_templates.md` for 12 ready-to-use templates

**Basic Templates:**

Protein Binder:
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

Small Molecule:
```yaml
entities:
  - protein:
      id: A
      sequence: 100..150
  - ligand:
      smiles: "c1ccccc1"
      id: L
```

Cyclic Peptide with Disulfide:
```yaml
entities:
  - protein:
      id: S
      sequence: 10..14C6C3
  - file:
      path: target.cif
      include:
        - chain:
            id: A
constraints:
  - bond:
      atom1: [S, 11, SG]
      atom2: [S, 18, SG]
```

#### YAML Key Rules

- File paths in YAML (`path:` field) are relative to the YAML file location
- Residue indices use `label_seq_id` (1-indexed), not `auth_seq_id`
- All chain IDs must be unique

#### Sequence Specification Formats

| Format | Meaning | Example |
|--------|---------|---------|
| `80..140` | Random length between 80-140 residues | `sequence: 80..140` |
| `80` | Exactly 80 designed residues | `sequence: 80` |
| `AAAVVV20PPP` | Specific residues with designed in middle | `sequence: AAAVVV20PPP` |
| `3..5C6C3` | Designed residues with specific cysteines | `sequence: 3..5C6C3` |

#### Save YAML Configuration

After confirming the YAML configuration with the user, save it to a file using the helper script:

```
script: "scripts/save_yaml_config.py"
args: "--yaml-text '<yaml_content>' --output-dir ./configs --protocol <protocol>"
```

**Example:**
```
script: "scripts/save_yaml_config.py"
args: "--yaml-text 'entities:
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
            binding: 45,67,89' --output-dir ./configs --protocol protein-anything"
```

Output: `./configs/boltzgen_config_protein_anything_<timestamp>.yaml`

**Script Options:**
| Option | Description |
|--------|-------------|
| `--yaml-text` | YAML configuration content (required) |
| `--output-dir` | Directory to save the file (default: `./configs`) |
| `--filename` | Custom filename (auto-generated if not provided) |
| `--protocol` | Protocol name for filename generation |
| `--validate` | Validate YAML format (default: True) |

### Step 2: Upload Files

Upload the saved YAML file and any CIF/PDB target files to the OpenBioMed server.

**Upload saved YAML file:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "./configs/boltzgen_config_<protocol>_<timestamp>.yaml"}'
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

### Step 3: Submit Design Job (Instant)

Use the `/run_pipeline/` endpoint to submit the BoltzGen design job. This returns immediately with a job_id.

**API call format:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_submit",
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

#### Response (Instant, < 1 second)

```json
{
  "task": "boltzgen_submit",
  "job_id": "abc123def456",
  "boltzgen_service_job_id": "87528178a91f",
  "status": "queued",
  "queue_position": 2,
  "boltzgen_service_url": "http://172.16.20.44:10002/jobs/87528178a91f",
  "message": "Job abc123def456 submitted to BoltzGen"
}
```

### Step 4: Start Background Monitoring

**⚠️ 重要提醒: BoltzGen 设计任务预计需要 12-45 分钟完成。**

后台监控每 2 分钟检查一次状态。您可以随时使用 `boltzgen_status` 查询当前进度。

**API call format:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_monitor",
  "job_id": "<job_id>"  // optional, monitors all active jobs if not provided
}
```

#### Response (Instant)

```json
{
  "task": "boltzgen_monitor",
  "monitoring": ["abc123def456"],
  "poll_interval": 120,
  "estimated_duration": "12-45 minutes",
  "message": "Background monitoring started. Design typically takes 12-45 minutes."
}
```

**Note:** The monitoring runs in background and automatically updates the job status in SQLite every 2 minutes. You can proceed with other tasks while waiting.

### Step 5: Check Job Status (Anytime)

Query the current job status from local SQLite (fast response, < 100ms).

**API call format:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_status",
  "job_id": "<job_id>"
}
```

#### Response

```json
{
  "task": "boltzgen_status",
  "job_id": "abc123def456",
  "status": "running",  // pending | queued | running | succeeded | failed | cancelled
  "progress": 45,
  "error_message": null,
  "boltzgen_service_url": "http://172.16.20.44:10002/jobs/87528178a91f"
}
```

### Step 6: Download Results (When Status == succeeded)

When the job status becomes `succeeded`, download the results.

**API call format:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_download",
  "job_id": "<job_id>"
}
```

#### Response

```json
{
  "task": "boltzgen_download",
  "job_id": "abc123def456",
  "output_files": [
    "./tmp/boltzgen/abc123def456/output/design.cif",
    "./tmp/boltzgen/abc123def456/output/intermediate_designs_inverse_folded/aggregate_metrics_analyze.csv",
    ...
  ],
  "description": "BoltzGen Protein-anything design completed.\nJob ID: abc123def456\n..."
}

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

**Input:** Design a 100-150 residue protein that binds benzene.

**Step 1 — Generate and Save YAML Configuration:**

```
script: "scripts/save_yaml_config.py"
args: "--yaml-text 'entities:
  - protein:
      id: A
      sequence: 100..150
  - ligand:
      smiles: \"c1ccccc1\"' --output-dir ./configs --protocol protein-small_molecule"
```
→ Output: `./configs/boltzgen_config_protein_small_molecule_<timestamp>.yaml`

**Step 2 — Upload YAML to OpenBioMed server:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "./configs/boltzgen_config_protein_small_molecule_<timestamp>.yaml"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.yaml"}`

**Step 3 — Submit to run_pipeline (Instant):**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_submit",
  "boltzgen_yaml_file": "./tmp/uploads/<uuid>.yaml",
  "boltzgen_protocol": "protein-small_molecule",
  "boltzgen_num_designs": 10,
  "boltzgen_budget": 2
}
```
→ Response: `{"job_id": "abc123", "status": "queued", ...}`

**Step 4 — Start Background Monitoring:**

⚠️ **BoltzGen 设计任务预计需要 12-45 分钟完成。**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_monitor",
  "job_id": "abc123"
}
```
→ Response: `{"monitoring": ["abc123"], "poll_interval": 120, "estimated_duration": "12-45 minutes"}`

**Step 5 — Check Status (Anytime):**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_status",
  "job_id": "abc123"
}
```
→ Response: `{"status": "running", "progress": 45, ...}`

**Step 6 — Download Results (When succeeded):**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_download",
  "job_id": "abc123"
}
```
→ Response: `{"output_files": [...], "description": "..."}`

### Example 2: Protein Binder

**Input:** Design an 80-120 residue binder for chain A of a target protein.

**Step 1 — Generate and Save YAML Configuration:**

```
script: "scripts/save_yaml_config.py"
args: "--yaml-text 'entities:
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
            binding: 45,67,89' --output-dir ./configs --protocol protein-anything"
```
→ Output: `./configs/boltzgen_config_protein_anything_<timestamp>.yaml`

**Step 2 — Upload YAML and CIF to OpenBioMed server:**

Upload YAML:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "./configs/boltzgen_config_protein_anything_<timestamp>.yaml"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.yaml"}`

Upload CIF:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<target.cif_path>"}'
```
→ Response: `{"path": "./tmp/uploads/<uuid>.cif"}`

**Step 3 — Submit (Instant):**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_submit",
  "boltzgen_yaml_file": "./tmp/uploads/<uuid>.yaml",
  "boltzgen_protocol": "protein-anything",
  "boltzgen_num_designs": 10,
  "boltzgen_budget": 2,
  "boltzgen_cif_files": ["./tmp/uploads/<uuid>.cif"]
}
```
→ Response: `{"job_id": "def456", "status": "queued", ...}`

**Step 4 — Start Monitoring:**

⚠️ **BoltzGen 设计任务预计需要 12-45 分钟完成。**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_monitor",
  "job_id": "def456"
}
```

**Step 5 — Check Status & Download (when succeeded):**

Status check: `{"task": "boltzgen_status", "job_id": "def456"}`
Download: `{"task": "boltzgen_download", "job_id": "def456"}`

### Example 3: Cyclic Peptide with Disulfide

**Input:** Design a 10-14 residue cyclic peptide with cysteine constraints.

**Step 1 — Generate and Save YAML Configuration:**

```
script: "scripts/save_yaml_config.py"
args: "--yaml-text 'entities:
  - protein:
      id: S
      sequence: 10..14C6C3
  - file:
      path: target.cif
      include:
        - chain:
            id: A
constraints:
  - bond:
      atom1: [S, 11, SG]
      atom2: [S, 18, SG]' --output-dir ./configs --protocol peptide-anything"
```
→ Output: `./configs/boltzgen_config_peptide_anything_<timestamp>.yaml`

**Step 2 — Upload files:**

Upload YAML:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "./configs/boltzgen_config_peptide_anything_<timestamp>.yaml"}'
```

Upload CIF:
```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload"
method: "POST"
files: '{"file": "<target.cif_path>"}'
```

**Step 3 — Submit:**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {
  "task": "boltzgen_submit",
  "boltzgen_yaml_file": "./tmp/uploads/<uuid>.yaml",
  "boltzgen_protocol": "peptide-anything",
  "boltzgen_num_designs": 10,
  "boltzgen_budget": 2,
  "boltzgen_cif_files": ["./tmp/uploads/<uuid>.cif"]
}
```

**Step 4 — Monitor & Download:**

⚠️ **BoltzGen 设计任务预计需要 12-45 分钟完成。**

```
url: "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
method: "POST"
body: {"task": "boltzgen_monitor", "job_id": "<job_id>"}
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

## Reference Documents

| Document | Location | Content |
|----------|----------|---------|
| Conversational Guide | `references/parameters_guide.md` | Question templates by protocol, conversation examples |
| YAML Templates | `references/yaml_templates.md` | 12 complete configuration templates |
| Test Cases | `test_cases/*.yaml` | Example YAML configurations |

## Next Steps

Validate designs with **structure-prediction-boltz-2** skill to confirm predicted structures fold correctly.