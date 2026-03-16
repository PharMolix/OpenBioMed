---
name: mutation-design-aav
description: Propose high-fitness and high-diversity mutants of the VP1 capsid protein of Adeno-Associated Virus (AAV) through multi-round iterative optimization.
---

# High-fitness AAV Mutant Proposal

A skill performs automated multi-round optimization of a 28-amino acid segment of the VP1 capsid protein of Adeno-Associated Virus (AAV) to discover mutants with improved DNA packaging fitness and high sequence diversity.


## When to Use This Skill

* Design novel AAV mutants with improved DNA packaging fitness.
* Run computational iterative directed evolution.
* Perform fast mutation search guided by an oracle model.

Example prompts:

* “Design AAV mutants with higher DNA packaging fitness.”
* “Run multi-round mutation optimization for AAV.”
* “Generate 96 AAV variants with improved fitness.”


## Prerequisites

* Python 3.9+
* PyTorch
* NumPy / Pandas
* Protein sequence analysis tools
* Protein language model tools (ESM2)


## Core Capabilities

This skill can:

1. Download **initial AAV sequences** if they were not provided by users.
2. Download and execute an in-silico **oracle AAV prediction model**.
3. Generate controllable mutants within **4 point mutations** for each round.
4. Use **ESM2 embeddings** to represent protein sequences.
5. Optimize mutation proposals based on **oracle feedback**.
6. Maintain population **diversity using average pairwise Hamming distance**.
7. Perform **multi-round optimization** and return the best mutants.

## Workflow

1. Download initial AAV sequences from `https://cloud.tsinghua.edu.cn/f/992109032d8049689a6d/?dl=1` and use them as the starting pool.

2. Download the oracle AAV prediction model from `https://cloud.tsinghua.edu.cn/f/80bbc575ec3f4e63a0af/?dl=1`, and the configuration file from `https://cloud.tsinghua.edu.cn/f/09ea0869b74b4d2ca53e/?dl=1`.

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

6. Evaluation: evaluate all candidate sequences using the oracle scoring function. Use oracle feedback from previous rounds to bias mutation proposals toward directions that **increase predicted fitness** (fitness gradient exploitation).

7. Selection: rank sequences by predicted fitness and select the top 96 mutants, while **maintaining diversity** measured by average pairwise Hamming distance.

8. Repeat proposal, evaluation, and selection until 10 rounds are completed, or best fitness does not improve for 3 consecutive rounds.

9. Collect the best 96 mutants discovered across all rounds and sort them by predicted DNA packaging fitness, and export the results as a CSV file following the specified output format.



## Output Format

The final result must be a **CSV file** with two columns:

| sequence            | fitness                |
| ------------------- | ---------------------- |
| AAV_mutant_sequence | predicted_fitness      |

Requirements:

* Exactly **96 sequences**
* Sorted by **fitness in descending order**
* Sequences must be **valid AAV mutants**

Example:

```
sequence,fitness
ADMEIIQVNPYSSEQYGDVATPLYHGTG,0.96
ADMEIRQVNPYSSEQYGDVATPLQHGTG,0.93
ADSELASTNPVSTELYGIVATNLMAQAS,0.92
...
```

This CSV represents the **final optimized AAV mutant library** predicted to exhibit higher DNA packaging fitness.
