---
name: structure-prediction-boltz-2
description: >
  Structure prediction using Boltz-2, an open biomolecular structure predictor.
  Use this skill when:
  (1) Predicting protein complex structures,
  (2) Validating designed binders,
  (3) Predicting protein-ligand complexes,
  (4) Using local GPU resources.

  For protein complex binding affinity evaluation, use prodigy.
license: MIT
category: design-tools
tags: [structure-prediction, protein complex, protein-ligand complex]
---

# Boltz-2 Structure Prediction

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.10 |
| CUDA | 12.0+ | 12.2 |
| GPU VRAM | 24GB | 80GB (A800) |
| RAM | 32GB | 64GB |

## How to run

### Local installation
```bash
pip install boltz[cuda] -U -i
```

### Predict protein complex structure
```python
import os, yaml, subprocess

def predict_protein_complex_structure(sequence_1, sequence_2, project_dir):
    """
    :param sequence_1: sequence of the first protein
    :param sequence_2: sequence of the second protein
    :param project_dir: path to the project
    :return structure of the protein complex in PDB format
    """
    # init project dir
    os.makedirs(project_dir, exist_ok=True)
    log_file = os.join(project_dir, 'log.txt')
    # init input yaml file
    data = {
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence_1,
                    "msa": "empty"
                }
            },
            {
                "protein": {
                    "id": "B",
                    "sequence": sequence_2,
                    "msa": "empty"
                }
            },
        ]
    }
    input_file = os.path.join(project_dir, "input.yaml")
    with open(input_file, "w") as f:
        yaml.dump(data, f)
    # init output file
    output_dir = os.path.join(project_dir, "boltz")
    # prediction
    command = [
        'boltz', 'predict', input_file,
        "--out_dir", output_dir,
        '--use_msa_server',
        '--output_format', "pdb",
        "--seed", "42"
    ]
    with open(log_file, 'a') as f:
        process = subprocess.Popen(command, stdout=f, stderr=f, env=self.env)
        process.communicate()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill() 
        process.wait()

    # extract structure
    with open(os.path.join(output_dir, "boltz_results_input", "predictions", "input", "input_model_0.pdb"), 'r') as f:
        pred_struc = f.read()

    return pred_struc

# Predict protein complex structure for sequence_1 and sequence_2
# pred_structure is the structure prediction in PDB format
pred_structure = predict_protein_complex_structure(sequence_1, sequence_2, project_dir)
```

### Predict protein ligand complex structure and IC50

```python
import os, yaml, subprocess

def predict_protein_ligand_complex_affinity(sequence, smiles, project_dir):
    """
    :param sequence: sequence of the first protein
    :param smiles: SMILES string of the ligand
    :param project_dir: path to the project
    :return pred_struct: structure of protein-ligand complex in PDB format
    :return pred_ic50: binding affinity prediction of the protein-ligand complex
    """
    # init project dir
    os.makedirs(project_dir, exist_ok=True)
    log_file = os.join(project_dir, 'log.txt')
    # init input yaml file
    data = {
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "msa": "empty"
                }
            },
            {
                "ligand":{
                    "id": "B",
                    "smiles": smiles,
                }
            }
        ],
        "properties": [
            {
                "affinity":{
                    "binder": "B"
                }
            }
        ]
    }
    input_file = os.path.join(project_dir, "input.yaml")
    with open(input_file, "w") as f:
        yaml.dump(data, f)

    # init output file
    output_dir = os.path.join(project_dir, "boltz")
    # prediction
    command = [
        'boltz', 'predict', input_file,
        "--out_dir", output_dir,
        '--use_msa_server',
        '--output_format', "pdb",
        "--seed", "42"
    ]
    with open(log_file, 'a') as f:
        process = subprocess.Popen(command, stdout=f, stderr=f, env=self.env)
        process.communicate()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill() 
        process.wait()

    # extract affinity
    with open(os.path.join(output_dir, "boltz_results_input", "predictions", "input", "affinity_input.json"), 'r') as f:
        pred_ic50 = (6 - json.load(f)["affinity_pred_value"]) * 1.364
    # extract structure
    with open(os.path.join(output_dir, "boltz_results_input", "predictions", "input", "input_model_0.pdb"), 'r') as f:
        pred_struc = f.read()

    return pred_struc, pred_ic50

# Predict protein-ligand complex structure and the corresponding binding affinity (IC50)
# pred_structure is the structure prediction in PDB format
# pred_ic50 is the binding affinity prediction in IC50 format
pred_structure, pred_ic50 = predict_protein_ligand_complex_affinity(sequence, smiles, project_dir):
```

## Output format

### Protein complex structure prediction
```
project_dir/boltz/boltz_results_input/predictions/input/
├── input_model_0.pdb               # structure prediction (PDB format)
├── confidence_input_model_0.json   # pTM, ipTM
├── pae_input_model_0.npz           # PAE matrix
└── plddt_input_model_0.npz         # pLDDT matrix
```

### Protein-ligand complex structure prediction
```
project_dir/boltz/boltz_results_input/predictions/input/
├── input_model_0.pdb               # structure prediction (PDB format)
└── affinity_input.json             # affinity_pred_value
```

## Decision tree

```
Should I use Boltz-2?
│
└─ What are you predicting?
   ├─ Structure prediction for general protein-protein complex → boltz-2 ✓
   ├─ Structure prediction for protein-ligand complex → boltz-2 ✓
   ├─ Antibody and nanobody structure prediction → tfold
   └─ Antigen-antibody structure prediction → tfold
```

## Typical performance

| Campaign Size | Time (L40S) | Cost (Modal) | Notes |
|---------------|-------------|--------------|-------|
| 100 complexes | 30-45 min | ~$8 | Standard validation |
| 500 complexes | 2-3h | ~$35 | Large campaign |
| 1000 complexes | 4-6h | ~$70 | Comprehensive |

**Per-complex**: ~15-30s for typical binder-target complex.

**Next**: Evaluate protein complex binding affinity with `prodigy`.