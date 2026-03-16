# Workflow Notes

## Best Starting Point

For new users, start from `examples/finetune_integration.py` because it shows
the repo's intended sequence:

1. load AnnData
2. prepare batch metadata
3. load checkpoint and vocabulary
4. preprocess and bin
5. tokenize
6. train and evaluate

## Embedding Use Cases

Use `scgpt.tasks.cell_emb` when the user wants:

- `X_scGPT` representations for visualization or downstream models
- checkpoint-based embedding inference without reimplementing tokenization
- a reusable representation for reference mapping or batch-aware comparison

## Common Pitfalls

- vocabulary mismatch causing most genes to be dropped
- double log transform
- treating normalized values as raw counts
- forgetting that the model expects `<cls>` and padding conventions from the repo
- assuming a tutorial notebook is interchangeable with a packaged task function
