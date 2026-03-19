---
name: functional-protein-design
description: >
  Generate functional protein sequences using CodeFP. Use this skill when:
  (1) Generating de novo protein sequences guided by specific Gene Ontology (GO) tags.
  (2) Exploring sequence space with prior functional constraints.
license: MIT
category: design-tools
tags: [protein-design, go-guided, sequence-generation, structure-prediction]
---

# CodeFP Functional Protein Design

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Environment | Configured via OpenBioMed: [`README.md`](https://github.com/PharMolix/OpenBioMed/blob/main/README.md) |
| Hardware | CUDA-compatible GPU (≥ 10GB VRAM) required for both generation and folding |
| Checkpoints | Download CodeFP weights & mappings from [Google Drive](https://drive.google.com/drive/folders/1Zqp2uD-f3cSzXeg35ixK-Epf-HpKBQYY?usp=sharing). |

## Data Preparation & Configuration

**Directory Structure**
Organize your downloaded checkpoints and mapping files as follows:

```text
checkpoints/
├── codefp/
│   ├── model/
│   │   └── checkpoints/
│   │       └── model.ckpt
│   └── mappings/
│       ├── go_mapping.pkl
│       ├── go_id_mapping.pkl
│       ├── desc2map_dict_statics.pkl
│       └── train_go_terms_cls_emb.pkl

```

## How to Run

Before getting started, ensure that your environment is fully configured. This includes a successful installation of OpenBioMed and the completion of all required model weight downloads.

Next, search the [Gene Ontology website](https://geneontology.org/docs/ontology-documentation/) to identify 1–3 Molecular Function (MF) GO terms (e.g., ['GO:0004930', 'GO:0004984']) that best align with your functional target.

Note: Please ensure that the selected GO terms are included in go_mapping.pkl, a dictionary whose keys enumerate all supported GO terms (e.g., “GO:0004930”, “GO:0004984”), to ensure compatibility with the model.

### Python API

This approach allows for direct GO-guided generation and immediate folding into a PDB structure.

```python
from open_biomed.core.pipeline import InferencePipeline

# 1. GO-guided sequence generation
generator = InferencePipeline(
    task="go_guided_protein_generation",
    model="codefp",
    model_ckpt="./checkpoints/codefp/model/checkpoints/model.ckpt",
    device="cuda:0"
)

# Replace with 1-3 target Molecular Function (MF) GO terms
go_terms = [['GO:0004930', 'GO:0004984']] 
designed_seqs = generator.run(go_terms=go_terms) 
seq_only = designed_seqs[0][0]

# 2. Complete 3D structure via folding
folder = InferencePipeline(
    task="protein_folding",
    model="esmfold",
    model_ckpt="./checkpoints/server/esmfold.ckpt", # ESMFold will be downloaded automatically.
    device="cuda:0"
)

folded_protein = folder.run(protein=seq_only)

# 3. Save output
folded_protein[0][0].save_pdb("designed.pdb")

```

## Expected Deliverables

Every successful run must yield a report containing:

1. **The generated PDB file** (`designed.pdb`).
2. **The exact GO tags used** for generation.
3. **Brief descriptions** of each GO tag.

**Sample Deliverable Report:**

* **Used GO Terms:** `['GO:0004930', 'GO:0004984']`
* **Descriptions:** * `GO:0004930` — G protein-coupled receptor activity
* `GO:0004984` — Olfactory receptor activity



## Troubleshooting

| Error / Warning | Cause | Fix |
| --- | --- | --- |
| `Warning: "GO ID {go_id} not found in mapping, using hash instead."` | Target GO ID is not supported in `go_mapping.pkl`. | **1.** Find the closest alternative GO combination in `go_mapping.pkl`.<br>**2.** Rerun with the alternative.<br>**3.** Explicitly report the substituted GO combination to the user. |
| `FileNotFoundError` or Checkpoint fails to load | Incorrect paths in `codefp.yaml`. | Verify file paths match the actual directory structure. |