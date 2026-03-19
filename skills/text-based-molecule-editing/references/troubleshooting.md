# Troubleshooting Guide

Common issues and solutions for text-based molecule editing.

## Installation Issues

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'open_biomed'`

**Solution**: Install OpenBioMed in development mode:
```bash
cd /path/to/OpenBioMed
pip install -e .
```

### CUDA Not Available

**Error**: `AssertionError: Torch not compiled with CUDA support`

**Solution**: Use CPU mode (slower but functional):
```python
pipeline = InferencePipeline(
    task="text_based_molecule_editing",
    model="molt5",
    model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
    device="cpu"
)
```

## Model Loading Issues

### Checkpoint Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Verify checkpoint path exists:
```bash
ls -la ./checkpoints/server/text_based_molecule_editing_biot5.ckpt
```

Download from the OpenBioMed repository if missing.

### Base Model Not Found

**Error**: `LocalEntryNotFoundError` or HuggingFace download failure

**Solution**: Ensure MolT5 base model is available:
```bash
ls -la ./checkpoints/molt5/base/
```

If missing, download from HuggingFace or set `HF_ENDPOINT`:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Checkpoint Size Mismatch

**Error**: `RuntimeError: size mismatch` when loading weights

**Solution**: The checkpoint may be for a different model. Verify:
- Model config matches checkpoint (molt5 vs biot5)
- Checkpoint is for `text_based_molecule_editing` task

## Runtime Issues

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Use CPU: `device="cpu"`
2. Clear GPU cache before inference:
```python
import torch
torch.cuda.empty_cache()
```
3. Reduce batch size (if running multiple molecules)

### Invalid SMILES Generated

**Symptom**: `edited_molecule` is `None` or has empty SMILES

**Causes**:
- Model generated invalid SMILES syntax
- Text prompt was too ambiguous

**Solutions**:
1. Try a more specific prompt:
```python
# Vague (may fail)
text = "make it better"

# Specific (better results)
text = "This molecule should be more soluble in water"
```

2. Run multiple times with different seeds:
```python
import random
import numpy as np
import torch

for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    outputs = pipeline.run(molecule=molecule, text=text)
    if outputs[0][0] is not None:
        break
```

### Slow Inference

**Symptom**: Inference takes >10 seconds per molecule

**Causes**:
- Running on CPU
- Large molecule (long SMILES)

**Solutions**:
1. Use GPU: `device="cuda:0"`
2. Reduce beam search:
```python
# In config or model setup
model.config.predict.num_beams = 1  # Greedy decoding
```

## Input Issues

### Invalid SMILES Input

**Error**: `RDKit ERROR: [Explicit valence]` or similar

**Cause**: Input SMILES is chemically invalid

**Solution**: Validate SMILES before processing:
```python
from rdkit import Chem

smiles = "CC(=O)Oc1ccccc1C(=O)O"
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError(f"Invalid SMILES: {smiles}")
```

### Molecule Name Not Found

**Error**: `ValueError: Could not find molecule` from PubChem

**Solution**:
1. Try alternative names (IUPAC, brand names)
2. Provide SMILES directly:
```python
molecule = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
```

## Property Calculation Issues

### QED Calculation Fails

**Error**: `RuntimeError` during QED calculation

**Cause**: Molecule has unusual atoms or charges

**Solution**: Check molecule validity:
```python
from rdkit.Chem import QED

try:
    qed_score = QED.qed(molecule.rdmol)
except Exception as e:
    print(f"QED calculation failed: {e}")
    qed_score = None
```

### LogP Out of Range

**Symptom**: LogP value seems unrealistic (>10 or <-5)

**Cause**: Wild-Dman-Crippen method has limitations

**Solution**: Cross-validate with experimental data or alternative calculators.

## Output Issues

### No Meaningful Change

**Symptom**: Edited SMILES is identical or very similar to input

**Causes**:
- Prompt was too subtle
- Model didn't learn the property relationship

**Solutions**:
1. Use more explicit prompts:
```python
text = "Replace the methyl group with a hydroxyl group to increase solubility"
```

2. Try multiple edits in sequence:
```python
# First edit
text1 = "This molecule should be more soluble"
mol1 = pipeline.run(molecule=mol, text=Text.from_str(text1))[0][0]

# Second edit
text2 = "This molecule should have higher QED"
mol2 = pipeline.run(molecule=mol1, text=Text.from_str(text2))[0][0]
```

### Unexpected Structural Changes

**Symptom**: Edited molecule has dramatic unexpected changes

**Cause**: Model learned unexpected correlations in training data

**Solution**: Use more constrained prompts or post-filter results:
```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# Check similarity to original
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol.rdmol, 2, nBits=2048)
fp2 = AllChem.GetMorganFingerprintAsBitVect(edited.rdmol, 2, nBits=2048)
similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

if similarity < 0.3:
    print("Warning: Edited molecule is very different from original")
```
