---
name: mutation-design-gfp
description: Propose high-fluorescence and high-diversity mutants of Green Fluorescent Protein (GFP) through multi-round iterative optimization.
---

# High-Fluorescence GFP Mutant Proposal

A skill performs automated multi-round optimization of Green Fluorescent Protein (GFP) to discover mutants with higher fluorescence intensity and higher diversity.


## When to Use This Skill

* Design novel GFP mutants with improved fluorescence intensity.
* Run computational iterative directed evolution.
* Perform fast mutation search guided by an oracle model.

Example prompts:

* “Design GFP mutants with higher fluorescence.”
* “Run multi-round mutation optimization for GFP.”
* “Generate 96 GFP variants with improved fluorescence.”


## Prerequisites

* Python 3.9+
* PyTorch
* NumPy / Pandas
* Protein sequence analysis tools
* Protein language model tools (ESM2)


## Core Capabilities

This skill can:

1. Download **initial GFP sequences** if they were not provided by users.
2. Download and execute an in-silico **oracle GFP prediction model**.
3. Generate controllable mutants within **4 point mutations** for each round.
4. Use **ESM2 embeddings** to represent GFP sequences.
5. Optimize mutation proposals based on **oracle feedback**.
6. Maintain population **diversity using average pairwise Hamming distance**.
7. Perform **multi-round optimization** and return the best mutants.

## Workflow

1. Download initial GFP sequences from `https://cloud.tsinghua.edu.cn/f/5e673c1db710466b828f/?dl=1` and use them as the starting pool.

2. Download the oracle GFP prediction model from `https://cloud.tsinghua.edu.cn/f/f655f79d7bb04a98a0bb/?dl=1`, and the configuration file from `https://cloud.tsinghua.edu.cn/f/8a894bb4b41f4074b9b0/?dl=1`.

3. Execute code for oracle loading and scoring:
```py
import torch
from omegaconf import OmegaConf

# ===== ORACLE MODEL LOADING =====
def load_oracle_model(ckpt_path, cfg_path):
    with open(cfg_path, 'r') as fp:
        cfg = OmegaConf.load(fp.name)
    oracle = BaseCNN(**cfg.model.predictor)
    state_dict = torch.load(ckpt_path)
    oracle.load_state_dict(torch.load(ckpt_path))
    oracle.eval()

# ===== ORACLE SCORING FUNCTION =====
def score_sequence(oracle, sequence: str) -> float:
    results = oracle(sequence).detach()
    return results.cpu().numpy()
```

4. Compute ESM2 embeddings for all sequences to represent sequence features.

5. Proposal: for each round, propose 96 × 4 candidate mutants from the current population using only point mutations with ≤4 mutations per sequence.

6. Evaluation: evaluate all candidate sequences using the oracle scoring function. Use oracle feedback from previous rounds to bias mutation proposals toward directions that **increase predicted fluorescence** (fitness gradient exploitation).

7. Selection: rank sequences by predicted fitness and select the top 96 mutants, while **maintaining diversity** measured by average pairwise Hamming distance.

8. Repeat proposal, evaluation, and selection until 10 rounds are completed, or best fitness does not improve for 3 consecutive rounds.

9. Collect the best 96 mutants discovered across all rounds and sort them by predicted fluorescence, and export the results as a CSV file following the specified output format.



## Output Format

The final result must be a **CSV file** with two columns:

| sequence            | fitness                |
| ------------------- | ---------------------- |
| GFP_mutant_sequence | predicted_fluorescence |

Requirements:

* Exactly **96 sequences**
* Sorted by **fitness in descending order**
* Sequences must be **valid GFP mutants**

Example:

```
sequence,fitness
SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT...,0.93
SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFIATT...,0.91
SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTIKFICTT...,0.89
...
```

This CSV represents the **final optimized GFP mutant library** predicted to exhibit higher fluorescence intensity.
