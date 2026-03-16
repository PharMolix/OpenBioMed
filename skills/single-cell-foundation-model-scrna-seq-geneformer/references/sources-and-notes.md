# Sources And Notes

## Official Sources

- documentation: `https://geneformer.readthedocs.io/en/latest/`
- model repository / installation path: `https://huggingface.co/ctheodoris/Geneformer`

## Project Positioning

Geneformer is positioned around:

- transfer learning in network biology
- fine-tuning with limited task-specific data
- zero-shot or low-shot perturbation-style analysis
- representations that support cell-state and gene-network reasoning

## Practical Distinctions

- Geneformer tokenization is a first-class step, not just input formatting
- embeddings and perturbations are major supported workflows, not side utilities
- many downstream tasks operate on tokenized `.dataset` objects rather than `.h5ad`
