# Protein Mutation Analysis - Troubleshooting

## Common Issues

### 1. Model Checkpoint Not Found

**Symptom**: `FileNotFoundError` or `AttributeError: 'Struct' object has no attribute 'hf_model_name_or_path'`

**Cause**: Required model checkpoints are not at expected paths.

**Solution**: Check and create symlinks to checkpoints:

```bash
# Check existing checkpoints
ls -la ./checkpoints/

# Expected paths for MutaPLM:
# ./checkpoints/server/mutaplm.pth
# ./checkpoints/esm2/650m/
# ./checkpoints/biomedgpt-lm/

# If checkpoints are in a different location, create symlinks
ln -s /path/to/esm2-650m ./checkpoints/esm2/650m
ln -s /path/to/biomedgpt-lm ./checkpoints/biomedgpt-lm
```

### 2. Conda Environment Not Activated

**Symptom**: `ModuleNotFoundError: No module named 'open_biomed'`

**Solution**: Activate the correct conda environment:

```bash
conda activate biomed_dev
```

### 3. Visualization Fails with TypeError

**Symptom**: `TypeError: 'NoneType' object is not iterable`

**Cause**: The protein object doesn't have 3D coordinates for visualization.

**Solution**: Ensure structure prediction (ESMFold) completed successfully before visualization:

```python
# Correct order
folding_tool = TOOLS["protein_folding"]
fold_result, _ = folding_tool.run(protein=protein)
predicted_protein = fold_result.get("protein")

# Now visualize
viz_tool = TOOLS["visualize_protein"]
viz_result, _ = viz_tool.run(protein=predicted_protein)
```

### 4. Invalid Mutation Format

**Symptom**: `ValueError` or unexpected behavior

**Cause**: Mutation string doesn't follow the required format.

**Valid Format**: `OriginalAA + Position + MutantAA`

| Valid | Invalid | Reason |
|-------|---------|--------|
| R248Q | R248 | Missing mutant AA |
| R248Q | 248Q | Missing original AA |
| R248Q | ARG248GLN | Use single-letter codes |
| V600E | V600 | Missing mutant AA |

### 5. Position Out of Range

**Symptom**: Error during mutation analysis

**Cause**: Mutation position exceeds protein sequence length.

**Solution**: Verify position is within sequence bounds:

```python
def validate_mutation(protein, mutation):
    # Extract position from mutation string
    position = int(mutation[1:-1])  # "R248Q" -> 248

    if position < 1:
        raise ValueError(f"Position must be >= 1, got {position}")

    if position > len(protein.sequence):
        raise ValueError(
            f"Position {position} exceeds sequence length {len(protein.sequence)}"
        )

    # Verify original AA matches
    original_aa = mutation[0]
    actual_aa = protein.sequence[position - 1]  # 0-indexed

    if original_aa != actual_aa:
        raise ValueError(
            f"Expected {original_aa} at position {position}, "
            f"but found {actual_aa}"
        )

    return True
```

### 6. ESMFold Slow on Long Sequences

**Symptom**: Structure prediction takes very long time

**Cause**: ESMFold scales with sequence length.

**Approximate Times**:
- < 200 aa: ~2-5 seconds
- 200-500 aa: ~10-30 seconds
- 500-1000 aa: ~1-2 minutes
- > 1000 aa: May take several minutes

**Solution**: For very long proteins, consider:
- Using existing PDB structures if available
- Focusing on specific domains
- Using a GPU with more memory

## Alternative Approaches

### Web Search Fallback

If MutaPLM is unavailable, use web search:

```python
from open_biomed.tools.tool_registry import TOOLS

# Get protein name first
tool = TOOLS["protein_uniprot_request"]
result, _ = tool.run(accession="P04637")
protein = result.get("protein")

# Search for mutation information
web_tool = TOOLS["web_search"]
search_result, _ = web_tool.run(
    query=f"{protein.name} R248Q mutation effect"
)
```

### Using PDB Structures

If the protein has a known structure:

```python
from open_biomed.tools.tool_registry import TOOLS

# Retrieve from PDB
pdb_tool = TOOLS["protein_pdb_request"]
result, _ = pdb_tool.run(accession="1TUP")  # Example: TP53-DNA complex
protein = result.get("protein")

# Now you have experimental structure, no need for ESMFold
viz_tool = TOOLS["visualize_protein"]
viz_result, _ = viz_tool.run(protein=protein, style="cartoon")
```

## Performance Tips

1. **Reuse protein objects**: If analyzing multiple mutations on the same protein, retrieve it once and reuse
2. **Skip visualization**: For batch processing, use `--no-viz` flag
3. **Cache structures**: Save predicted structures with `export_protein` for later use
