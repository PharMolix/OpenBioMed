# Model Notes

## Official Sources

- repo: `https://github.com/PharMolix/SToFM`
- paper: `https://arxiv.org/abs/2507.11588`

## Paper-Level Positioning

The paper presents SToFM as a multi-scale foundation model for spatial
transcriptomics. The key idea is that ST data requires integrated modeling of:

- macro-scale tissue morphology
- micro-scale cellular microenvironment
- gene-scale expression

The arXiv abstract states that SToFM constructs multi-scale ST sub-slices and
then applies an SE(2) Transformer to obtain cell representations. The paper also
introduces `SToCorpus-88M`, described there as the largest high-resolution
spatial transcriptomics pretraining corpus at release time.

## Practical Interpretation

Operationally, the repo exposes a two-stage representation pipeline:

1. a transcriptome cell encoder that produces per-cell embeddings
2. a spatial SE(2) Transformer that refines them using multi-scale spatial structure

This means SToFM should be treated as a spatial representation model, not just a
Geneformer-style tokenizer plus a classifier.

## What To Emphasize In Use

When helping with SToFM, prioritize:

1. correct preprocessing into `hf.dataset` plus `data.h5ad`
2. correct handling of spatial coordinates
3. correct checkpoint wiring for both encoder stages
4. embedding generation before downstream task heads
