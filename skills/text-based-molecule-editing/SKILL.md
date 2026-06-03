---
name: text-based-molecule-editing
description: |
  Modify molecules based on natural language descriptions using MolT5/BioT5 models.
  Use this skill when:
  (1) User wants to modify a molecule to improve specific properties (solubility, potency, etc.),
  (2) User provides a molecule and asks to "make it more X" or "improve Y",
  (3) User wants to generate molecule variants guided by text descriptions.
  Triggers on phrases like "modify this molecule", "edit the molecule", "make it more soluble",
  "improve drug-likeness", "change the molecule to", "optimize this compound".
license: MIT
category: drug-discovery
tags: [molecule-editing, text-guided, molecular-optimization, de-novo-design]
---

# Text-Based Molecule Editing

Modify molecular structures guided by natural language property descriptions, via the OpenBioMed server API.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://127.0.0.1:8092`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `http://localhost:9000` or `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: if the molecule data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/run_pipeline/` or `${OPENBIOMED_API_BASE_URL}/web_search/`.

## When to Use

- User wants to optimize a molecule for specific properties (solubility, binding, drug-likeness)
- User provides a molecule and requests property-based modifications
- User wants to explore structural variants guided by text descriptions

## Workflow

### Step 1: Get the Molecule SMILES (if user provides a name)

Only needed when the user gives a molecule name instead of a SMILES string. If the user already provides a SMILES, skip this step.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "<molecule_name>"}'
```

Response:
```json
{
  "task": "molecule_name_request",
  "molecule": "<PubChem data>",
  "molecule_preview": "<SMILES string>"
}
```

Extract the `molecule_preview` field — this is the SMILES string for subsequent steps.

### Step 2: Calculate Baseline Properties (Optional)

Compare properties before and after editing. Two options depending on server configuration:

**Option A: `molecule_property_calculation`** — supports QED, LogP, Lipinski. Available on self-hosted servers with full pipeline support.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_calculation", "molecule": "<SMILES>", "property": "QED"}'
```

Response:
```json
{"task": "molecule_property_calculation", "model": "...", "score": "<QED value>"}
```

Replace `"property"` with `LogP` or `Lipinski` for other properties.

**Option B: `molecule_property_prediction`** — predicts BBBP penetration, SIDER side effects, etc. Available on all servers.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "<SMILES>", "dataset": "BBBP"}'
```

Response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp", "score": "The blood-brain barrier penetration of the molecule is [0.188]"}
```

Supported datasets: `BBBP` (blood-brain barrier), `SIDER` (side effects), `caco2_wang` (Caco-2 permeability), `half_life_obach` (half-life), `ld50_zhu` (LD50 toxicity).

### Step 3: Run Text-Based Editing

Call the molecule editing endpoint with the SMILES and the natural language edit description:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "text_based_molecule_editing", "model": "biot5", "molecule": "<SMILES>", "text": "<edit description>"}'
```

Response:
```json
{
  "task": "text_based_molecule_editing",
  "model": "biot5",
  "molecule": "<path to edited molecule file>",
  "molecule_preview": "<edited SMILES string>"
}
```

**Note**: The `molecule` field contains a file path on the server. External agents cannot access this path directly. Use Step 4 below to get the molecule content.

### Step 4: Read Molecule File Content (Recommended for External Agents)

After receiving the molecule file path, call `read_molecule_file` to get the actual molecule content:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "read_molecule_file", "molecule": "<molecule_file_path>", "value": "true"}'
```

Response:
```json
{
  "task": "read_molecule_file",
  "smiles": "<edited SMILES string>",
  "name": "<molecule name>",
  "sdf_content": "<SDF file content for 3D structure>",
  "description": "Molecule content read from <file_path>: SMILES=<smiles>"
}
```

**Parameters**:
- `molecule`: The molecule file path from Step 3 response
- `value`: "true" to include SDF content (3D structure), "false" for SMILES only

### Step 5: Compare Properties

Re-calculate properties for the edited molecule using the same method from Step 2 and compare with baseline values.

**Option A** (molecule_property_calculation):
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_calculation", "molecule": "<edited_SMILES>", "property": "QED"}'
```

**Option B** (molecule_property_prediction):
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "<edited_SMILES>", "dataset": "BBBP"}'
```

Summarize the before/after comparison.

## Expected Outputs

| Step | API Endpoint | Response Field | Output |
|------|-------------|---------------|--------|
| 1 (optional) | `/web_search/` | `molecule_preview` | Original SMILES |
| 2 (optional) | `/run_pipeline/` | `score` | Baseline property values |
| 3 | `/run_pipeline/` | `molecule` | Edited molecule file path |
| 3 | `/run_pipeline/` | `molecule_preview` | Edited SMILES (preview) |
| 4 | `/run_pipeline/` | `smiles`, `sdf_content` | Full molecule content accessible to external agents |
| 5 (optional) | `/run_pipeline/` | `score` | New property values for comparison |

## Interpretation Guide

### LogP (Lipophilicity)

| Value | Solubility | Interpretation |
|-------|------------|----------------|
| < 0 | High water solubility | Very hydrophilic |
| 0-2 | Moderate | Good balance for oral drugs |
| 2-5 | Low water solubility | May need formulation help |
| > 5 | Very lipophilic | Poor absorption likely |

### QED (Quantitative Estimate of Drug-likeness)

| Value | Quality | Interpretation |
|-------|---------|----------------|
| > 0.7 | Excellent | Highly drug-like |
| 0.5-0.7 | Good | Acceptable drug-likeness |
| 0.3-0.5 | Moderate | May need optimization |
| < 0.3 | Poor | Significant liabilities |

## Example Usage

**Input**: "Make aspirin more soluble in water"

**Step 1**: Get aspirin SMILES
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}'
```

Expected response:
```json
{"task": "molecule_name_request", "molecule": "...", "molecule_preview": "CC(=O)Oc1ccccc1C(=O)O"}
```

**Step 2**: Calculate baseline BBBP penetration
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "dataset": "BBBP"}'
```

Expected response:
```json
{"task": "molecule_property_prediction", "model": "graphmvp", "score": "The blood-brain barrier penetration of the molecule is [0.188]"}
```

**Step 3**: Edit molecule
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "text_based_molecule_editing", "model": "biot5", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "text": "This molecule should be more soluble in water"}'
```

Expected response:
```json
{"task": "text_based_molecule_editing", "model": "biot5", "molecule": "./tmp/...", "molecule_preview": "CC(=O)Oc1ccc(C(=O)O)cc1C(=O)O"}
```

**Step 4**: Read molecule file content (for external agents)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "read_molecule_file", "molecule": "./tmp/..."}'
```

Expected response:
```json
{"task": "read_molecule_file", "smiles": "CC(=O)Oc1ccc(C(=O)O)cc1C(=O)O", "sdf_content": "..."}
```

**Step 5**: Calculate BBBP for edited molecule
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "molecule_property_prediction", "model": "graphmvp", "molecule": "CC(=O)Oc1ccc(C(=O)O)cc1C(=O)O", "dataset": "BBBP"}'
```

Compare baseline vs. edited BBBP scores to evaluate the edit effect.

## Model Options

The `text_based_molecule_editing` task supports multiple models via the `model` field:

| Model | Description |
|-------|-------------|
| `biot5` (default) | BioT5 model for biomedical molecule editing |
| `molt5` | MolT5 model specialized for molecules |
| `biot5_plus` | BioT5+ enhanced model |

To use a different model, change the `model` field in the Step 3 curl command.

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Molecule Name Not Found

**Symptom**: `/web_search/` returns empty or null `molecule_preview`.

**Solution**: Ask user for the SMILES string directly and skip Step 1.

### Invalid SMILES Output

**Symptom**: `/run_pipeline/` returns empty or invalid `molecule_preview`.

**Solution**:
- Rephrase the edit prompt
- Try a different model (e.g., `molt5` instead of `biot5`)
- Run multiple times for different outputs

## Notes

- The edited molecule may not always perfectly satisfy the edit description — it is a model-predicted variant
- Running the edit multiple times can produce different structural variants
- Property comparison (Step 4) is optional but recommended to verify the edit achieved the desired effect