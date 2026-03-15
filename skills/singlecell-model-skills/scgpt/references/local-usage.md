# Local Usage

## Repo Root

`/DATA/disk0/zhaosy/home/scGPT`

## Installation Notes

The local README recommends:

```bash
pip install scgpt "flash-attn<1.0.5"
```

`flash-attn` is optional but recommended. The repo notes CUDA compatibility can matter.

## Core Local Entry Points

| Goal | Entry point |
|---|---|
| preprocess data | `scgpt/preprocess.py` |
| extract embeddings | `scgpt/tasks/cell_emb.py` |
| fine-tune for integration | `examples/finetune_integration.py` |
| tutorial-driven workflow | `tutorials/*.ipynb` |

## Expected Checkpoint Structure

Most pretrained workflow code expects a model directory containing:

- `args.json`
- `vocab.json`
- `best_model.pt`

If one of these files is missing, stop and fix the checkpoint path first.

## Typical scGPT Preprocessing Pattern

The integration example uses:

- `normalize_total=1e4`
- optional `log1p`
- HVG selection
- `binning=n_bins`
- storage in `adata.layers["X_binned"]`

It also prepares:

- `adata.obs["str_batch"]`
- `adata.obs["batch_id"]`
- `adata.var["gene_name"]`

## Embedding Extraction Pattern

`scgpt.tasks.cell_emb.embed_data(...)`:

- accepts an AnnData object or `.h5ad`
- loads the checkpoint files from `model_dir`
- filters genes by vocabulary membership
- encodes cells
- writes `X_scGPT` embeddings to `adata.obsm` or returns a new AnnData

## Practical Checks

Before running scGPT, verify:

1. the checkpoint folder contains `args.json`, `vocab.json`, and `best_model.pt`
2. the chosen gene column really matches the vocabulary namespace
3. the required layer or matrix is non-negative before binning
4. batch labels are present if the selected workflow uses them
5. the AnnData object still contains enough matched genes after vocabulary filtering
