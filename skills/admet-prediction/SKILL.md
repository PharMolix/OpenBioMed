---
name: admet-prediction
description: >
  Predict comprehensive ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
  properties for drug candidate molecules using GraphMVP ensemble models.
  Use this skill when:
  (1) Predicting blood-brain barrier penetration,
  (2) Assessing side effect profiles,
  (3) Estimating Caco-2 permeability, half-life, or LD50 toxicity,
  (4) Evaluating drug-likeness and safety of molecules.
license: MIT
category: admet-prediction
tags: [admet, toxicity, drug-discovery, pharmacokinetics, graphmvp]
---

# ADMET Prediction

Predict comprehensive ADMET properties for drug candidate molecules using GraphMVP ensemble models.

## When to Use

- User asks to predict ADMET properties for a molecule
- User provides a drug candidate and wants safety assessment
- User needs blood-brain barrier penetration prediction
- User wants to evaluate toxicity (LD50) or side effects (SIDER)
- User requests pharmacokinetic properties (half-life, Caco-2)

## Workflow

### Step 1: Load Molecule

Create molecule from SMILES string.

```python
from open_biomed.data import Molecule
molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
```

### Step 2: Build ADMET Pipeline

Initialize ensemble pipeline with all GraphMVP checkpoints.

```python
from open_biomed.core.pipeline import InferencePipeline, EnsemblePipeline

pipelines = {
    "BBBP": InferencePipeline(
        task="molecule_property_prediction", model="graphmvp",
        model_ckpt="./checkpoints/server/graphmvp-BBBP.ckpt",
        additional_config="./configs/dataset/bbbp.yaml", device="cuda:0"),
    "SIDER": InferencePipeline(
        task="molecule_property_prediction", model="graphmvp",
        model_ckpt="./checkpoints/server/graphmvp-SIDER.ckpt",
        additional_config="./configs/dataset/sider.yaml", device="cuda:0"),
    # See examples/basic_example.py for full pipeline setup
}
pipeline = EnsemblePipeline(pipelines)
```

### Step 3: Run Predictions

Execute predictions for each ADMET property.

```python
# BBB penetration
bbb_result = pipeline.run(molecule=molecule, task="BBBP")

# Side effects (27 categories)
sider_result = pipeline.run(molecule=molecule, task="SIDER")

# Regression properties
caco2_result = pipeline.run(molecule=molecule, task="caco2_wang")
half_life_result = pipeline.run(molecule=molecule, task="half_life_obach")
ld50_result = pipeline.run(molecule=molecule, task="ld50_zhu")
```

## Expected Outputs

| Task | Output Type | Description |
|------|-------------|-------------|
| BBBP | float [0-1] | Probability of BBB penetration |
| SIDER | list[27 floats] | Side effect probabilities per category |
| caco2_wang | float | Log permeability (cm/s) |
| half_life_obach | float | Log half-life (hours) |
| ld50_zhu | float | Log LD50 (mg/kg) |

## Interpretation Guide

### BBB Penetration

| Value | Interpretation |
|-------|----------------|
| > 0.5 | Likely crosses BBB |
| < 0.5 | Unlikely to cross BBB |

### Caco-2 Permeability

| Value (log cm/s) | Interpretation |
|------------------|----------------|
| > -5 | High absorption |
| -6 to -5 | Moderate absorption |
| < -6 | Low absorption |

### LD50 Toxicity

| Value (log mg/kg) | Toxicity Level |
|-------------------|----------------|
| < 1 | Highly toxic (<10 mg/kg) |
| 1-2 | Moderately toxic (10-100 mg/kg) |
| 2-3 | Slightly toxic (100-1000 mg/kg) |
| > 3 | Low toxicity (>1000 mg/kg) |

### SIDER Side Effects

Values range 0-1. Categories with **> 0.7** indicate high risk of that side effect.

## Error Handling

### Checkpoint Not Found

**Symptom**: `FileNotFoundError: graphmvp-*.ckpt`

**Solution**: Ensure checkpoints exist in `./checkpoints/server/`:
```bash
ls checkpoints/server/graphmvp-*.ckpt
```

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Use CPU instead:
```python
# Change device from "cuda:0" to "cpu"
device="cpu"
```

### Invalid SMILES

**Symptom**: Molecule fails to parse

**Solution**: Validate SMILES format or use molecule name lookup via PubChem.

## Example

```
Input: aspirin (CC(=O)OC1=CC=CC=C1C(=O)O)

Output:
  BBB Penetration: 0.19 (does NOT cross BBB)
  Caco-2: -4.68 (moderate absorption)
  Half-life: -7.06 (short half-life)
  LD50: 2.06 (moderate toxicity ~115 mg/kg)

  Top Side Effects:
    Skin disorders: 0.80
    Nervous system: 0.78
    Gastrointestinal: 0.78
```

## See Also

- `examples/basic_example.py` - Full runnable example with all properties
- `references/sider_categories.md` - Complete SIDER category list
- `references/interpretation.md` - Detailed interpretation guidelines
