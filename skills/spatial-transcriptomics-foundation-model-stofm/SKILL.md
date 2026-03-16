---
name: single-cell-foundation-model-stofm
description: >
  Use this skill when a task involves the local SToFM project in
  /DATA/disk0/zhaosy/home/SToFM, especially preprocessing spatial
  transcriptomics data for SToFM, generating cell embeddings with the cell
  encoder plus SE(2) Transformer pipeline, handling spatial coordinates, or
  preparing SToFM embeddings for downstream region segmentation or cell type
  annotation.
---

# SToFM

## Use This Skill When

Use this skill for the local SToFM repository at `/DATA/disk0/zhaosy/home/SToFM`.
It is the right choice when the task involves:

- preprocessing spatial transcriptomics data into the format expected by SToFM
- converting mouse genes to the human Geneformer vocabulary when needed
- generating cell embeddings with the official `get_embeddings.py` pipeline
- working with spatial coordinates, sub-slice splitting, or hypernode construction
- using SToFM embeddings for downstream region segmentation or cell type annotation
- understanding the repo's two-stage architecture: cell encoder plus SE(2) Transformer

Do not use this skill for ordinary scRNA-seq analysis without spatial coordinates.

## Start Here

1. Confirm the data has usable spatial coordinates.
2. Check whether the input has already been preprocessed into both `data.h5ad` and `hf.dataset`.
3. Check that the required checkpoints exist for both the cell encoder and the SE(2) Transformer.
4. Prefer the official embedding pipeline before building downstream heads.

## Choose A Path

### Preprocessing

Use `preprocessing/preprocess.py` first unless the dataset is already in the
expected SToFM format.

The repo's preprocessing flow:

- starts from `AnnData`
- expects Geneformer-style transcriptome tokenization
- adds `obs["n_counts"]`
- uses `var["ensembl_id"]`
- maps mouse gene ids to human ids when needed
- saves both:
  - `hf.dataset` for the cell encoder
  - `data.h5ad` for later spatial loading

### Embedding generation

The main official workflow is `get_embeddings.py`.

This path:

- loads the pretrained cell encoder
- loads the SToFM SE(2) Transformer
- encodes cells from `hf.dataset` if `ce_emb.npy` is missing
- loads spatial coordinates from `data.h5ad`
- splits large slices into sub-slices
- builds hypernodes and attention biases
- runs the SE(2) Transformer
- saves final embeddings such as `stofm_emb.npy`

### Downstream tasks

The repo's recommended downstream pattern is simple:

1. generate SToFM embeddings first
2. train a task-specific head on top of those embeddings

The README specifically highlights:

- tissue region semantic segmentation
- cell type annotation

## Guardrails

- Do not skip spatial information; SToFM is not just a transcriptome encoder.
- Do not treat `hf.dataset` alone as sufficient input for the full model; the SE(2) stage also needs spatial structure.
- Do not assume mouse genes can be used directly; the repo explicitly maps them into the human vocabulary.
- Do not run downstream heads on raw expression if the intended workflow is SToFM; generate official embeddings first.
- Do not assume the repo ships checkpoints; they are external downloads.

## Read More Only If Needed

- Read `references/local-usage.md` for paths, required files, and practical checks.
- Read `references/model-notes.md` for the paper-level positioning and the multi-scale workflow.
