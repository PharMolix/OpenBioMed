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

Modify molecular structures guided by natural language property descriptions.

## When to Use

- User wants to optimize a molecule for specific properties (solubility, binding, drug-likeness)
- User provides a molecule and requests property-based modifications
- User wants to explore structural variants guided by text descriptions

## Workflow

### Step 1: Prepare Input Molecule

```python
from open_biomed.data import Molecule
from open_biomed.tools.tool_registry import TOOLS

# Option A: From molecule name (queries PubChem)
tool = TOOLS["molecule_name_request"]
result, _ = tool.run(accession="aspirin")
molecule = result[0]

# Option B: From SMILES directly
molecule = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
```

### Step 2: Calculate Baseline Properties (Optional)

```python
qed_tool = TOOLS["molecule_qed"]
logp_tool = TOOLS["molecule_logp"]
sa_tool = TOOLS["molecule_sa"]

qed, _ = qed_tool.run(molecule=molecule)
logp, _ = logp_tool.run(molecule=molecule)
sa, _ = sa_tool.run(molecule=molecule)
```

### Step 3: Run Text-Based Editing

```python
from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Text

pipeline = InferencePipeline(
    task="text_based_molecule_editing",
    model="molt5",
    model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
    device="cuda:0"
)

outputs = pipeline.run(
    molecule=molecule,
    text=Text.from_str("This molecule should be more soluble in water"),
)
edited_molecule = outputs[0][0]
```

### Step 4: Compare Properties

```python
qed_new, _ = qed_tool.run(molecule=edited_molecule)
logp_new, _ = logp_tool.run(molecule=edited_molecule)

print(f"Original SMILES: {molecule.smiles}")
print(f"Edited SMILES: {edited_molecule.smiles}")
print(f"LogP change: {logp[0]:.2f} → {logp_new[0]:.2f}")
```

## Expected Outputs

| Step | Output | Description |
|------|--------|-------------|
| Step 1 | `Molecule` object | Input molecule with SMILES |
| Step 2 | `float` values | QED (0-1), LogP, SA scores |
| Step 3 | `Molecule` object | Edited molecule with new structure |
| Step 4 | Comparison | Before/after property summary |

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

### SA (Synthetic Accessibility)

| Value | Difficulty | Interpretation |
|-------|------------|----------------|
| 1-3 | Easy | Straightforward synthesis |
| 3-5 | Moderate | Some challenges |
| 5-7 | Difficult | Complex synthesis needed |
| > 7 | Very difficult | Likely impractical |

## Error Handling

### Model Checkpoint Not Found

**Symptom**: `FileNotFoundError` for checkpoint file

**Solution**: Ensure checkpoint exists at `./checkpoints/server/text_based_molecule_editing_biot5.ckpt`

```python
import os
ckpt_path = "./checkpoints/server/text_based_molecule_editing_biot5.ckpt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Download checkpoint to: {ckpt_path}")
```

### Invalid SMILES Output

**Symptom**: Model generates invalid SMILES string

**Solution**: The model returns `None` for invalid molecules. Try:
- Rephrasing the edit prompt
- Using beam search with more beams
- Running multiple times for different outputs

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Use CPU or smaller batch:

```python
pipeline = InferencePipeline(
    task="text_based_molecule_editing",
    model="molt5",
    model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
    device="cpu"  # Fallback to CPU
)
```

## Example

```
Input: aspirin
Prompt: "This molecule should be more soluble in water"

Original SMILES: CC(=O)Oc1ccccc1C(=O)O
Edited SMILES:   CC(=O)Oc1ccc(C(=O)O)cc1C(=O)O

Property Changes:
  LogP: 1.31 → 1.01 (-0.30, more soluble)
  QED:  0.55 → 0.59 (+0.04, better drug-likeness)
  SA:   1.58 → 1.81 (+0.23, slightly harder to synthesize)
```

## See Also

- `examples/basic_example.py` - Full runnable example script
- `examples/solubility_optimization.py` - Solubility-focused workflow
- `references/troubleshooting.md` - Detailed error handling
- `references/advanced.md` - Advanced prompt engineering tips
