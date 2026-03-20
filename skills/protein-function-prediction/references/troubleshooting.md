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
3. Focusing on specific regions of interest

```python
# Process N-terminal domain (first 500 residues)
protein_nterm = Protein.from_fasta(sequence[:500])

# Process C-terminal domain
protein_cterm = Protein.from_fasta(sequence[-500:])
```

### Empty or Generic Output

**Issue:** Model returns generic/uninformative response.

**Possible causes:**
- Sequence is too short (< 50 residues)
- Sequence contains non-standard amino acids
- Model hasn't seen similar sequences during training

**Solution:**
Try rephrasing the question or check sequence validity:
```python
# Validate sequence contains only standard amino acids
import re
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
sequence = "".join([aa for aa in sequence.upper() if aa in valid_aa])
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

### Slow Inference on CPU

**Issue:** Inference takes > 30 seconds on CPU.

**Solutions:**
1. Use GPU if available
2. Batch multiple sequences together
3. Consider using a smaller model or quantization

## Getting Help

If issues persist:
1. Check the OpenBioMed documentation
2. Review the example scripts in `examples/`
3. Open an issue on GitHub with:
   - Full error message
   - Input sequence (or first 50 residues)
   - Environment details (Python version, GPU/CPU)