# Advanced Usage

## Custom Questions

You can customize the question to get specific types of information:

```python
# For enzyme classification
question = Text.from_str("What enzyme class does this protein belong to?")

# For pathway information
question = Text.from_str("What biological pathway is this protein involved in?")

# For structural features
question = Text.from_str("What are the key structural features of this protein?")

# For disease associations
question = Text.from_str("Is this protein associated with any diseases?")
```

## Batch Processing

For annotating multiple proteins:

```python
from open_biomed.data import Protein, Text
from open_biomed.core.pipeline import InferencePipeline

# Load pipeline once
pipeline = InferencePipeline(
    task="protein_question_answering",
    model="biot5",
    model_ckpt="./checkpoints/server/protein_question_answering_biot5.ckpt",
    device="cuda:0"
)

question = Text.from_str(
    "Inspect the protein sequence and offer a concise description of its properties."
)

# Process multiple sequences
sequences = [
    "MRVGVIRFPGSNCDRDVHHVLELAGAEPEYVWW...",
    "MTENPVKKQLQDNRLYDFLGDEIYTRNIQSLLKD...",
    # ... more sequences
]

results = []
for seq in sequences:
    protein = Protein.from_fasta(seq)
    output = pipeline.run(protein=protein, text=question)
    results.append(output[0][0].str)
```

## Integration with UniProt

Combine with UniProt query for enriched annotations:

```python
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.data import Text
from open_biomed.core.pipeline import InferencePipeline

# Get protein from UniProt
uniprot_tool = TOOLS["protein_uniprot_request"]
protein, _ = uniprot_tool.run(accession="P00533")  # EGFR

# Get AI-predicted function
pipeline = InferencePipeline(
    task="protein_question_answering",
    model="biot5",
    model_ckpt="./checkpoints/server/protein_question_answering_biot5.ckpt",
    device="cuda:0"
)

question = Text.from_str(
    "Inspect the protein sequence and offer a concise description of its properties."
)
output = pipeline.run(protein=protein, text=question)
print(output[0][0].str)
```

## Combining with Structure Prediction

Get both function prediction and structure:

```python
from open_biomed.data import Protein, Text
from open_biomed.core.pipeline import InferencePipeline

sequence = "YOUR_SEQUENCE"
protein = Protein.from_fasta(sequence)

# Function annotation
qa_pipeline = InferencePipeline(
    task="protein_question_answering",
    model="biot5",
    model_ckpt="./checkpoints/server/protein_question_answering_biot5.ckpt",
    device="cuda:0"
)
question = Text.from_str(
    "Inspect the protein sequence and offer a concise description of its properties."
)
function = qa_pipeline.run(protein=protein, text=question)

# Structure prediction
fold_pipeline = InferencePipeline(
    task="protein_folding",
    model="esmfold",
    model_ckpt="./checkpoints/server/esmfold.ckpt",
    device="cuda:0"
)
structure = fold_pipeline.run(protein=protein)

print(f"Function: {function[0][0].str}")
print(f"Structure saved to: {structure[1][0]}")
```

## Output Interpretation

The model output typically contains:

| Information Type | How to Identify |
|------------------|-----------------|
| Enzyme name | Usually in the first sentence |
| EC classification | May include EC number format (e.g., EC 6.3.5.3) |
| Pathway | Look for "pathway", "biosynthesis", "metabolism" |
| Complex membership | Mentions of "subunit", "complex", "component" |
| Catalytic mechanism | Describes substrates, products, cofactors |
| Cellular location | "membrane", "cytoplasmic", "nuclear" |

## Model Performance

| Metric | Value |
|--------|-------|
| Sequence length limit | 512 amino acids |
| Inference time (GPU) | ~2-3 seconds |
| Inference time (CPU) | ~10-15 seconds |
| Memory requirement | ~4GB GPU VRAM |
