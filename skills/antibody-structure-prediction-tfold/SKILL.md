---
name: antibody-structure-prediction-tfold
description: >
  Antibody-related structure prediction using tfold model.
  Use this skill when:
  (1) Predict antibody and nanobody structure of a given sequence,
  (2) Predict antigen-antibody complex structure of given sequences,
  (3) Using local GPU resources.

  For binding affinity evaluation, use prodigy.
license: MIT
category: design-tools
tags: [structure-prediction, antibody, nanobody, antigen-antibody complex]
---

# tFold Antibody-related Structure Prediction

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.8 |
| CUDA | 11.7+ | 11.8 |
| GPU VRAM | 24GB | 80GB (A800) |
| RAM | 32GB | 64GB |

## How to run

### Local installation
```bash
git clone https://github.com/TencentAI4S/tfold.git
cd tfold

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install deepspeed==0.12.3 termcolor==2.3.0 biopython==1.79 ml-collections==0.1.1 dm-tree==0.1.8 numpy==1.21.2 modelcif==0.9 scipy requests
```

### Predict structure of an antibody
```python
import torch
import tfold

def pred_antibody_structure(heavy_chain_sequence, light_chain_sequence, output_path):
    """
    :param heavy_chain_sequence: sequence of the heavy chain
    :param light_chain_sequence: sequence of the light chain
    :param output_path: path to the antibody structure prediction
    """

    # Download the pre-trained model
    ppi_model_path = tfold.model.esm_ppi_650m_ab()
    tfold_model_path = tfold.model.tfold_ab_trunk()

    # Load the model
    model = tfold.deploy.PLMComplexPredictor.restore_from_module(ppi_model_path, tfold_model_path)

    # Prepare antibody sequences (can be single or multiple sequences)
    data =[
            {
              "sequence": heavy_chain_sequence, # Heavy chain
              "id": 'H'
              },
            {
              "sequence": light_chain_sequence, # Light chain
              "id": 'L'
              }]

    model.infer_pdb(data, output_path)
```

### Predict the structure of antigen-antibody complex
```python
import torch
import tfold
from projects.tfold_ag.gen_msa import generate_msa

def pred_antigen_antibody_structure(antigen_sequence, heavy_chain_sequence, light_chain_sequence, output_path):
    """
    :param antigen_sequence: sequence of the antigen
    :param heavy_chain_sequence: sequence of the heavy chain
    :param light_chain_sequence: sequence of the light chain
    :param output_path: path to the antibody structure prediction
    """

    # Download the pre-trained model of ESM-PPI
    ppi_model_path = tfold.model.esm_ppi_650m_ab()
    # Download the pre-trained model of alphaFold
    alphafold_path  = tfold.model.alpha_fold_4_ptm()
    # Download base model for tFold-Ag
    tfold_model_path = tfold.model.tfold_ag_base()

    # Load the model
    model = tfold.deploy.AgPredictor(ppi_model_path, alphafold_path, tfold_model_path)

    # generate msa information
    with open('antigen.fasta', 'w') as f:
      f.write(f'>antigen\n{antigen_sequence}')
    generate_msa('antigen.fasta', output_dir='./')
    with open('./antigen.a3m') as f:
      msa, deletion_matrix = tfold.protein.parser.parse_a3m(f.read())

    # prepare input
    data = [
            {
                "id": "H",
                "sequence": heavy_chain_sequence
            },
            {
                "id": "L",
                "sequence": light_chain_sequence
            },
            {
                "id": "A",
                "sequence": antigen_sequence,
                "msa": msa,
                "deletion_matrix": deletion_matrix
            }
            ]

    model.infer_pdb(data, output_path)
```

## Decision tree

```
Should I use tFold?
│
└─ What are you predicting?
   ├─ Antibody and nanobody structure prediction → antibody-structure-prediction-tfold ✓
   ├─ Antigen-antibody structure prediction → antibody-structure-prediction-tfold ✓
   ├─ Structure prediction for general protein-protein complex → structure-prediction-boltz-2
   └─ Structure prediction for protein-ligand complex → structure-prediction-boltz-2
```

**Next**: Evaluate binding affinity with `binding-affinity-prediction-prodigy`.