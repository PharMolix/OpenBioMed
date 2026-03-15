# Workflows

## Official Documentation

Geneformer documentation: `https://geneformer.readthedocs.io/en/latest/`

## Installation Pattern

The official docs describe installation from the model repository:

```bash
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

## Tokenizer Requirements

`TranscriptomeTokenizer` expects:

- raw counts scRNA-seq input
- `.loom` or `.h5ad`
- gene attribute `ensembl_id`
- cell attribute `n_counts`

Typical usage:

```python
from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ"}, nproc=4)
tk.tokenize_data("data_directory", "output_directory", "output_prefix")
```

## Classification

`Classifier` is used on tokenized `.dataset` inputs, not raw AnnData.
It supports:

- cell classification
- gene classification
- cross-validation / evaluation workflows

The docs explicitly describe cell-state and gene-classification inputs separately.

## Embeddings

`EmbExtractor` supports:

- `emb_mode="cls"`
- `emb_mode="cell"`
- `emb_mode="gene"`

It can also:

- plot cell embeddings
- generate state embedding dictionaries for perturbation workflows

## In Silico Perturbation

`InSilicoPerturber` is the primary perturbation API. Important arguments include:

- `perturb_type`
- `genes_to_perturb`
- `model_type`
- `emb_mode`
- `cell_states_to_model`
- `state_embs_dict`

Use this path when the goal is counterfactual cell-state movement rather than ordinary classification.

## Practical Checks

Before running Geneformer, verify:

1. the input still contains all genes detected in the transcriptome and was not feature-selected
2. `ensembl_id` and `n_counts` are present
3. the downstream script expects tokenized `.dataset` input rather than AnnData
4. the chosen model type matches the downstream artifact
5. perturbation tasks have clearly defined start and goal states
