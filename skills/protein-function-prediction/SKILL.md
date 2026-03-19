---
name: protein-function-prediction
description: >
  Predict protein function and properties from amino acid sequence using BioT5.
  Use this skill when:
  (1) You have a protein sequence and want to understand its biological function,
  (2) You need to identify enzyme activity, pathway involvement, or molecular interactions,
  (3) You want a concise description of protein properties from sequence alone.
license: MIT
category: protein-engineering
tags: [protein, function-prediction, annotation, sequence-analysis, biot5]
---

# Protein Function Prediction

Predict functional annotations and properties for proteins from their amino acid sequences using the BioT5 model.

## When to Use

- You have a protein FASTA sequence and need to understand its biological role
- You want to identify enzyme function, pathway involvement, or molecular mechanisms
- You need quick functional insights without experimental data
- You're characterizing novel or unannotated protein sequences

## Workflow

```python
from open_biomed.data import Protein, Text
from open_biomed.core.pipeline import InferencePipeline

# Create protein from FASTA sequence
protein = Protein.from_fasta("YOUR_AMINO_ACID_SEQUENCE")

# Create the question for functional annotation
question = Text.from_str(
    "Inspect the protein sequence and offer a concise description of its properties."
)

# Load the BioT5 model for protein question answering
pipeline = InferencePipeline(
    task="protein_question_answering",
    model="biot5",
    model_ckpt="./checkpoints/server/protein_question_answering_biot5.ckpt",
    device="cuda:0"
)

# Run inference to get functional annotation
outputs = pipeline.run(protein=protein, text=question)
function_description = outputs[0][0].str
print(function_description)
```

See `examples/basic_example.py` for a complete runnable script.

## Expected Outputs

The model returns a text description that typically includes:

| Output Component | Example |
|------------------|---------|
| **Enzyme name** | Phosphoribosylformylglycinamidine synthase |
| **Biological pathway** | Purine biosynthesis pathway |
| **Catalytic activity** | FGAR to FGAM conversion |
| **Complex membership** | Part of FGAM synthase complex (PurQ, PurL, PurS) |
| **Mechanism details** | ATP-dependent, glutamine amidotransferase activity |

### Example Output

> Part of the phosphoribosylformylglycinamidine synthase complex involved in the purines biosynthetic pathway. Catalyzes the ATP-dependent conversion of formylglycinamide ribonucleotide (FGAR) and glutamine to yield formylglycinamidine ribonucleotide (FGAM) and glutamate.

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
- **Capabilities**: Function prediction, property description, pathway annotation

## Limitations

- Sequences longer than 512 residues are truncated
- Model trained on known proteins; novel folds may have lower accuracy
- Does not predict 3D structure or binding sites (use `protein_folding` or `protein_binding_site_prediction` tools)

## Related Skills

- `protein-structure-design-boltzgen`: For 3D structure prediction
- `protein-mutation-analysis`: For mutation effect prediction
- `uniprot-query`: For retrieving protein metadata from UniProt
