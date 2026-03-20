---
name: single-cell-foundation-model-langcell
description: >
  Use this skill when a task involves the LangCell project for single-cell
  language-cell modeling, especially zero-shot cell type annotation, few-shot
  annotation, LangCell-CE finetuning, Geneformer-style tokenization, or
  preparing text descriptions for candidate cell identities and multimodal
  cell-text matching workflows.
---

# LangCell

## Use This Skill When

Use this skill for the local LangCell project at `/DATA/disk0/zhaosy/home/LangCell`.
It is the right choice when the task involves:

- zero-shot cell identity or cell type annotation from tokenized single-cell data
- few-shot cell type annotation with very limited labels
- finetuning only the LangCell cell encoder (`LangCell-CE`)
- preprocessing AnnData into the tokenized format expected by LangCell
- preparing text descriptions for candidate cell identities
- understanding how LangCell combines cell embeddings and text embeddings

Do not use this skill for ordinary Scanpy analysis that does not depend on LangCell.

## Start Here

1. Confirm whether the user wants zero-shot annotation, few-shot annotation, or cell-encoder-only finetuning.
2. Check that the input is already tokenized, or route through preprocessing first.
3. Check whether the required external assets exist: checkpoints, tokenized dataset, ontology / text-description JSON.
4. Prefer the zero-shot path first if the user is exploring LangCell rather than benchmarking a supervised baseline.

## Choose A Path

### Zero-shot annotation

Start here for most LangCell usage. The defining behavior is:

- encode cells with `cell_bert + cell_proj`
- encode candidate texts with `text_bert + text_proj`
- score cell-text matches with `ctm_head`
- combine similarity and matching scores for final predictions

Use `LangCell-annotation-zeroshot/zero-shot.ipynb` as the primary reference path.

### Few-shot annotation

Use `LangCell-annotation-fewshot/fewshot.py` when only a tiny labeled support set
is available and the user still wants the multimodal LangCell path.

### LangCell-CE finetuning

Use `LangCell-CE-annotation/finetune.py` when the user wants a standard
supervised classifier on top of the pretrained cell encoder.

### Preprocessing

LangCell does not take raw `.h5ad` directly for these downstream scripts. First:

- read AnnData with `scanpy`
- add `obs["n_counts"]`
- ensure `var["ensembl_id"]` exists
- tokenize with `LangCellTranscriptomeTokenizer`
- save with `save_to_disk(...)`

## Guardrails

- Do not claim raw `.h5ad` can be passed directly into LangCell inference; tokenization is required first.
- Do not assume checkpoints live in the repo. The official repo expects external downloads.
- Do not treat zero-shot prediction as plain nearest-neighbor on cell embeddings; the project combines similarity and cell-text matching.
- Do not assume label columns are named consistently. The repo checks several alternatives such as `celltype`, `cell_type`, `str_labels`, and `labels`.
- If new cell types are needed, prepare textual descriptions carefully instead of inventing bare labels with no ontology context.

## Read More Only If Needed

- Read `references/local-usage.md` for entry points, asset requirements, and practical checks.
- Read `references/model-notes.md` when the task depends on LangCell's multimodal design or paper-level motivation.
