---
name: protein-mutation-analysis
description: |
  Analyze protein mutations by retrieving protein data, explaining mutation effects,
  predicting protein structure, and visualizing results. Use this skill when the user
  asks about protein mutations, wants to understand mutation effects, or needs to
  analyze genetic variants. Triggers on phrases like "analyze mutation", "explain mutation",
  "what does this mutation do", "protein variant analysis".
---

# Protein Mutation Analysis

Analyze the functional impact of protein mutations using MutaPLM and visualize protein structures.

## When to Use

- User provides a UniProt ID and mutation (e.g., "P04637 R248Q")
- User wants to understand the effect of a specific mutation
- User needs to visualize a mutated protein structure
- Research on disease-associated genetic variants

## Workflow

### Step 1: Retrieve Protein from UniProt

```python
from open_biomed.tools.tool_registry import TOOLS

tool = TOOLS["protein_uniprot_request"]
result, message = tool.run(accession="P04637")
protein = result.get("protein")
```

### Step 2: Explain Mutation with MutaPLM

```python
mutation_tool = TOOLS["mutation_explanation"]
mutation_result, _ = mutation_tool.run(
    protein=protein,
    mutation="R248Q"  # Format: OriginalAA + Position + MutantAA
)
```

### Step 3: Predict Structure with ESMFold

```python
folding_tool = TOOLS["protein_folding"]
fold_result, _ = folding_tool.run(protein=protein)
predicted_protein = fold_result.get("protein")
```

### Step 4: Visualize Protein Structure

```python
viz_tool = TOOLS["visualize_protein"]
viz_result, _ = viz_tool.run(protein=predicted_protein, style="cartoon")
```

See `examples/basic_analysis.py` for the complete implementation.

## Expected Outputs

| Step | Output | Description |
|------|--------|-------------|
| Retrieve Protein | Protein object | Name, sequence from UniProt |
| Explain Mutation | Text | Functional impact from MutaPLM |
| Predict Structure | Protein with 3D coords | Structure from ESMFold |
| Visualize | PNG file | Rendered protein structure |

## Mutation Format

Single amino acid mutation: `OriginalAA + Position + MutantAA`

| Valid | Invalid | Reason |
|-------|---------|--------|
| R248Q | R248 | Missing mutant AA |
| V600E | 248Q | Missing original AA |
| L858R | ARG248GLN | Use single-letter codes |

## Error Handling

### Missing Model Checkpoints

**Symptom**: `FileNotFoundError` or `AttributeError`

**Solution**: Check checkpoints exist:
- `./checkpoints/server/mutaplm.pth`
- `./checkpoints/esm2/650m/`
- `./checkpoints/biomedgpt-lm/`

**Fallback**: Use web search for mutation literature.

### Position Out of Range

```python
position = int(mutation[1:-1])
if position > len(protein.sequence):
    print(f"Error: Position exceeds sequence length")
```

See `references/troubleshooting.md` for detailed error handling.

## Interpretation

### MutaPLM Output

- **Disease association**: "In [cancer type]..." indicates known disease link
- **Functional change**: Describes altered protein function
- **Structural impact**: May mention stability effects

### ESMFold Confidence

| pLDDT Score | Confidence |
|-------------|------------|
| > 90 | High |
| 70-90 | Moderate |
| < 70 | Low (disordered) |

## Example

```
Input: P04637 R248Q

Step 1: Retrieved TP53 (393 aa)
Step 2: "In lung cancer, mutation R248Q..."
Step 3: Structure predicted (~8s)
Step 4: Visualization saved

Output: Mutation analysis + structure + visualization
```

## Prerequisites

Model checkpoints required (see `references/troubleshooting.md`):
- MutaPLM, ESM2, BioMedGPT-LM, ESMFold

## Related Tools

- `protein_pdb_request` - Get existing PDB structures
- `protein_question_answering` - Ask about protein function
- `export_protein` - Save structure to PDB format
