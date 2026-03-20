---
name: protein-subcellular-localization-prediction-biot5
description: >
  Predict protein subcellular localization from amino acid sequence using BioT5.
  Use this skill when:
  (1) You have a protein sequence and want to know where it localizes in the cell,
  (2) You need to identify cellular compartment (nucleus, cytoplasm, membrane, etc.),
  (3) You want quick localization prediction without experimental data.
license: MIT
category: protein-engineering
tags: [protein, subcellular-localization, localization, sequence-analysis, biot5]
---

# Protein Subcellular Localization Prediction

Predict subcellular localization for proteins from their amino acid sequences using the BioT5 model.

## When to Use

- You have a protein FASTA sequence and need to know its cellular location
- You want to identify if a protein is cytoplasmic, nuclear, membrane-bound, or secreted
- You need quick localization insights without experimental data
- You're characterizing novel or unannotated protein sequences

## Workflow

```python
from open_biomed.data import Protein, Text
from open_biomed.core.pipeline import InferencePipeline

# Create protein from FASTA sequence
protein = Protein.from_fasta("YOUR_AMINO_ACID_SEQUENCE")

# Create the question for subcellular localization
question = Text.from_str(
    "Please provide information about the subcellular localization of this protein."
)

# Load the BioT5 model for protein question answering
pipeline = InferencePipeline(
    task="protein_question_answering",
    model="biot5",
    model_ckpt="./checkpoints/server/protein_question_answering_biot5.ckpt",
    device="cuda:0"
)

# Run inference to get localization prediction
outputs = pipeline.run(protein=protein, text=question)
localization = outputs[0][0].str
print(localization)
```

See `examples/basic_example.py` for a complete runnable script.

## Expected Outputs

The model returns subcellular localization information:

| Output | Description |
|--------|-------------|
| **Cytoplasm** | Cytoplasmic proteins, soluble enzymes |
| **Nucleus** | Nuclear proteins, transcription factors |
| **Membrane** | Membrane-bound proteins, receptors |
| **Secreted** | Extracellular proteins, secreted factors |
| **Mitochondria** | Mitochondrial proteins |
| **Peroxisome** | Peroxisomal enzymes |
| **Endoplasmic reticulum** | ER-resident proteins |
| **Golgi apparatus** | Golgi-localized proteins |

### Example Output

> Cytoplasm

## Input Formats

The skill accepts protein sequences in FASTA format (amino acid string):

```python
# From raw sequence string
protein = Protein.from_fasta("MRVGVIRFPGSNCDRDVHHVLELAGAEPEYVWW...")

# From UniProt (get sequence first)
from open_biomed.tools.tool_registry import TOOLS
tool = TOOLS["protein_uniprot_request"]
protein, _ = tool.run(accession="P00533")  # Example: EGFR
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Model checkpoint not found | Download checkpoint to `./checkpoints/server/` |
| `CUDA out of memory` | GPU memory insufficient | Use smaller batch or CPU device |
| `Sequence too long` | Exceeds 512 amino acid limit | Truncate sequence or use sliding window |

## Model Details

- **Model**: BioT5 (protein-text foundation model)
- **Max sequence length**: 512 amino acids
- **Inference time**: ~2-3 seconds per sequence on GPU
- **Capabilities**: Subcellular localization prediction

## Limitations

- Sequences longer than 512 residues are truncated
- Model trained on known proteins; novel sequences may have lower accuracy
- Single localization prediction (does not handle multi-localized proteins well)

## Related Skills

- `protein-function-annotation`: For function prediction
- `protein-mutation-analysis`: For mutation effect prediction
- `uniprot-query`: For retrieving protein metadata from UniProt
