---
name: scgpt
description: >
  Use this skill when a task involves the local scGPT project in
  /DATA/disk0/zhaosy/home/scGPT, especially scGPT preprocessing and binning,
  checkpoint vocabulary matching, cell embedding extraction, reference mapping,
  fine-tuning scGPT for integration or annotation, or using scGPT tutorials for
  GRN, perturbation, multiomics, and reference mapping workflows.
---

# scGPT

## Use This Skill When

Use this skill for the local scGPT repository at `/DATA/disk0/zhaosy/home/scGPT`.
It is the right choice when the task involves:

- preparing AnnData inputs with scGPT's own preprocessing pipeline
- matching genes to a pretrained scGPT vocabulary
- tokenizing binned expression inputs for transformer models
- extracting cell embeddings with pretrained checkpoints
- fine-tuning scGPT for integration or annotation-style downstream tasks
- understanding how scGPT expects binned values, special tokens, and batch labels
- working through scGPT tutorials such as integration, annotation, GRN, perturbation, or reference mapping

Do not use this skill for generic Scanpy work that does not depend on scGPT checkpoints or tokenization.

## Start Here

1. Confirm the checkpoint directory contains `args.json`, `vocab.json`, and `best_model.pt`.
2. Decide whether the task is fine-tuning, embedding extraction, or tutorial-guided experimentation.
3. Run preprocessing before tokenization or embedding unless the input has already been prepared for scGPT.
4. Check vocabulary overlap before spending time on training or inference.

## Choose A Path

### Preprocess and bin

The core preprocessing path in this repo is `scgpt.preprocess.Preprocessor`.
Typical steps include:

- filter genes by counts
- optionally filter cells
- normalize total counts
- optionally log1p transform
- subset highly variable genes
- bin values into discrete bins and store them in `adata.layers["X_binned"]`

### Fine-tune for integration

The clearest end-to-end example in the local repo is
`examples/finetune_integration.py`. It demonstrates:

- loading a dataset
- building `str_batch` and `batch_id`
- preprocessing and HVG selection
- matching checkpoint vocabulary
- tokenizing and padding batches
- training / evaluation for an integration workflow

If the user asks "how should I use scGPT on my AnnData?", this example is often
the best starting point.

### Extract cell embeddings

Use `scgpt.tasks.cell_emb.embed_data(...)` or related functions when the goal is
to generate cell embeddings from a pretrained model directory.

This path:

- loads `args.json`, `vocab.json`, and `best_model.pt`
- filters genes to those found in the vocabulary
- builds the transformer model
- encodes cells and writes embeddings to `adata.obsm["X_scGPT"]`

### Reference mapping or tutorial workflows

The local repo ships tutorials for:

- annotation
- integration
- multiomics
- GRN
- perturbation
- reference mapping

Use them when the user wants the project-supported path instead of building a
custom pipeline from scratch.

## Guardrails

- Do not assume arbitrary gene identifiers will work. scGPT inference depends on the checkpoint vocabulary.
- Do not skip binning if the target workflow expects `X_binned`.
- Do not mix raw, normalized, and logged layers casually; keep track of which layer is used at each stage.
- Do not assume all checkpoints use the same configuration; always read `args.json`.
- When extracting embeddings, verify the `gene_col` used to map genes into the vocabulary.

## Read More Only If Needed

- Read `references/local-usage.md` for checkpoint expectations, preprocessing shape, and local entry points.
- Read `references/workflow-notes.md` for best starting paths and common mistakes.
