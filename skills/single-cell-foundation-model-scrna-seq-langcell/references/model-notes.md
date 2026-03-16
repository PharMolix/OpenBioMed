# Model Notes

## Why LangCell Matters

LangCell was introduced as a language-cell pretraining framework for cell
identity understanding. Its central claim is that ordinary single-cell pretrained
models operate only on transcriptomics and therefore lack explicit identity
knowledge carried by text labels and ontology descriptions.

LangCell addresses this by aligning cells and texts in a shared representation
space and adding a cell-text matching objective.

## Main Practical Takeaways

- zero-shot annotation is the most distinctive capability
- textual descriptions are part of the model interface, not an optional afterthought
- the model is best suited for identity-oriented tasks such as cell type or related semantic annotation
- the official repo is lightweight and workflow-oriented, with some assets hosted externally

## Inference Components

The official implementation uses:

- a cell encoder (`cell_bert`)
- a text encoder (`text_bert`)
- projection layers for both modalities
- a cell-text matching head (`ctm_head`)

This means the prediction path is more than just embedding similarity.

## What To Emphasize In Use

When helping with LangCell, prioritize:

1. text-description quality
2. correct tokenization pipeline
3. correct checkpoint wiring
4. clear distinction between zero-shot multimodal use and cell-encoder-only finetuning
