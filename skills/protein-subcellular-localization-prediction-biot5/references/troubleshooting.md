# Troubleshooting

## Common Issues

### Model Checkpoint Not Found

**Error:**
```
FileNotFoundError: ./checkpoints/server/protein_question_answering_biot5.ckpt
```

**Solution:**
Download the BioT5 checkpoint to the server checkpoints directory:
```bash
# Checkpoint should be placed at:
./checkpoints/server/protein_question_answering_biot5.ckpt
```

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use CPU instead:
   ```python
   pipeline = InferencePipeline(..., device="cpu")
   ```
2. Reduce batch size or process one sequence at a time
3. Use a GPU with more VRAM

### Sequence Too Long

**Issue:** Sequences longer than 512 amino acids are truncated.

**Solution:**
For long sequences, consider:
1. Processing domains separately if known
2. Using a sliding window approach
3. Focusing on signal peptides or localization signals

```python
# Focus on N-terminal region (often contains localization signals)
protein_nterm = Protein.from_fasta(sequence[:500])
```

### Generic or Unclear Output

**Issue:** Model returns unexpected or generic localization.

**Possible causes:**
- Sequence lacks clear localization signals
- Model hasn't seen similar sequences during training
- Multi-localized protein (model may predict one location)

**Solution:**
Try additional questions for clarification:
```python
# Ask about membrane domains
question = Text.from_str("Does this protein have transmembrane domains?")

# Ask about signal peptides
question = Text.from_str("Does this protein have a signal peptide?")
```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'open_biomed'
```

**Solution:**
Ensure OpenBioMed is installed and in the Python path:
```bash
pip install -e .
# Or add to path in script:
import sys
sys.path.insert(0, "/path/to/OpenBioMed")
```

## Getting Help

If issues persist:
1. Check the OpenBioMed documentation
2. Review the example scripts in `examples/`
3. Open an issue on GitHub with:
   - Full error message
   - Input sequence (or first 50 residues)
   - Environment details (Python version, GPU/CPU)