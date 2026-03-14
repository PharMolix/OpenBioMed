---
name: geneformer
description: >
  Use this skill when a task involves Geneformer workflows, especially
  TranscriptomeTokenizer input preparation, tokenized `.dataset` generation,
  cell or gene classification with `Classifier`, embedding extraction with
  `EmbExtractor`, and in silico perturbation analysis with
  `InSilicoPerturber`.
---

# Geneformer

## Use This Skill When

Use this skill when the task involves official Geneformer workflows such as:

- converting raw scRNA-seq data into Geneformer tokenized datasets
- fine-tuning Geneformer for cell or gene classification
- extracting cell or gene embeddings
- generating state embeddings for downstream perturbation analysis
- running in silico perturbation or in silico treatment style analyses
- distinguishing pretrained zero-shot usage from fine-tuned classifier usage

This skill is for Geneformer-specific workflows, not generic single-cell model use.

## Start Here

1. Confirm the input is raw-count scRNA-seq data and still suitable for tokenization.
2. Check that `ensembl_id` and `n_counts` are available.
3. Tokenize first unless the user already has a Geneformer `.dataset`.
4. Decide whether the task is classification, embedding extraction, or in silico perturbation.

## Choose A Path

### Tokenization

Use `TranscriptomeTokenizer` first for almost every Geneformer workflow.
This step converts raw-count `.loom` or `.h5ad` data into tokenized datasets
used by the downstream APIs.

Geneformer expects:

- row attribute `ensembl_id`
- cell attribute `n_counts`

Optional metadata can be passed through during tokenization.

### Classification

Use `Classifier` for:

- cell state classification
- cell type annotation
- gene classification tasks

The input is a tokenized Geneformer `.dataset` object, not raw AnnData.

### Embedding extraction

Use `EmbExtractor` when the task is to:

- extract CLS, cell, or gene embeddings
- plot or inspect cell embeddings
- generate state embeddings for later perturbation analysis

### In silico perturbation

Use `InSilicoPerturber` for zero-shot or model-based perturbation analyses such as:

- deleting or shifting genes
- modeling start and goal cell states
- ranking perturbations by movement toward a desired cell state

This is one of Geneformer's defining workflows and should be treated as more
than ordinary classifier inference.

## Guardrails

- Do not pass feature-selected matrices into the tokenizer; the docs expect raw counts without feature selection.
- Do not use gene symbols where the tokenizer expects `ensembl_id`.
- Do not confuse tokenized `.dataset` files with AnnData objects.
- Do not skip tokenization and jump directly to classifiers or perturbation APIs.
- For perturbation tasks, be explicit about model type, embedding mode, and target cell states.

## Official Workflow Surface

| Component | Use |
|---|---|
| `TranscriptomeTokenizer` | create tokenized datasets |
| `Classifier` | fine-tune cell or gene classifiers |
| `MTLClassifier` | multitask cell classification |
| `EmbExtractor` | extract and summarize embeddings |
| `InSilicoPerturber` | simulate perturbations / treatment directions |

## Read More Only If Needed

- For operational usage and required input fields, read `references/workflows.md`.
- For official source locations and model-specific notes, read `references/sources-and-notes.md`.
