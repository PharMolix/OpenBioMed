# Local Usage

## Repo Root

`/DATA/disk0/zhaosy/home/SToFM`

## Installation Notes

The local repo pins:

- `torch==2.0.1`
- `transformers==4.39.1`
- `datasets==2.14.0`
- `scanpy==1.9.3`
- `rapids_singlecell==0.10.10`
- `cupy-cuda12x==13.3.0`
- `geneformer==0.0.1`

The README notes that `geneformer==0.0.1` is no longer on PyPI, so the repo
vendors a local `geneformer_001` copy. It also notes that `rapids_singlecell`
and `cupy` must match the machine's CUDA version.

## Required External Assets

The repo does not ship the pretrained checkpoints. The workflow typically needs:

- a cell encoder directory containing `cell_bert` and `cell_proj.bin`
- an SToFM config path
- an SToFM model checkpoint path

If these are missing, stop and ask for the real paths.

## Main Entry Points

| Goal | Entry point |
|---|---|
| preprocess AnnData for SToFM | `preprocessing/preprocess.py` |
| generate official embeddings | `get_embeddings.py` |
| spatial graph utilities | `model/extraction.py` |

## Preprocessing Pattern

`preprocessing/preprocess.py` does the following:

- subclasses `TranscriptomeTokenizer` as `SToFMTranscriptomeTokenizer`
- filters to genes in the Geneformer vocabulary
- uses `obs["n_counts"]`
- uses `var["ensembl_id"]`
- optionally maps mouse ids to human ids with `mouseid2humanid.pkl`
- writes:
  - `hf.dataset`
  - `data.h5ad`

Minimal output expectation for one dataset directory:

```text
dataset_dir/
├── data.h5ad
└── hf.dataset/
```

## Embedding Pipeline

`get_embeddings.py` expects dataset roots that contain:

- `data.h5ad`
- `hf.dataset`
- optionally `ce_emb.npy` if cell encoder embeddings were already generated

For each dataset root it:

1. runs `encode_cell(...)` if `ce_emb.npy` is missing
2. loads spatial data and cell embeddings
3. splits large slices into manageable sub-slices
4. clusters cells into hypernodes
5. builds graph inputs and attention bias matrices
6. runs the SE(2) Transformer
7. saves final embeddings, default `stofm_emb.npy`

## Practical Checks

Before running SToFM, verify:

1. the dataset has usable spatial coordinates in `adata.obsm["spatial"]` or an equivalent key
2. the dataset root contains both `data.h5ad` and `hf.dataset`
3. checkpoint paths for both stages are present
4. CUDA-compatible `cupy` and `rapids_singlecell` versions are installed
5. mouse data has been mapped to the human vocabulary before tokenization
