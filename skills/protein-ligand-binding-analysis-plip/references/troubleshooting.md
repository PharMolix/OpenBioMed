# Troubleshooting

## Common Issues and Solutions

### 1. No Ligands Found

**Problem:** PLIP reports no ligands in the PDB file.

**Causes & Solutions:**
- **Ligand not in HETATM records:** Ensure the ligand is defined as HETATM, not ATOM
- **Ligand excluded by MW threshold:** Lower `min_mw` parameter if ligand is small
- **Non-standard residue naming:** Check if ligand is properly defined in PDB header

```python
# Debug: Print all detected ligands
complex = PDBComplex()
complex.load_pdb(pdb_file)
for lig in complex.ligands:
    print(f"Found: {lig.hetid}, MW: {lig.mol.molwt}")
```

### 2. Characterization Fails

**Problem:** `complex.characterize_complex(lig)` raises an error.

**Solutions:**
- Check ligand has proper coordinates
- Verify ligand is not fragmented across multiple chains
- Try alternative PDB source (RCSB, PDBe)

```python
# Handle characterization errors gracefully
for lig in ligands:
    try:
        complex.characterize_complex(lig)
    except Exception as e:
        print(f"Failed to characterize {lig.hetid}: {e}")
        continue
```

### 3. PyMOL Visualization Issues

**Problem:** `visualize_in_pymol` fails or produces no output.

**Solutions:**
- Ensure PyMOL is installed: `pip install pymol-open-source`
- Check PyMOL is in system PATH
- Verify `config.OUTPATH` directory exists and is writable

```python
# Verify PyMOL installation
import subprocess
result = subprocess.run(["pymol", "-cq"], capture_output=True)
if result.returncode != 0:
    print("PyMOL not properly installed")
```

### 4. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'plip'`

**Solution:**
```bash
pip install plip
```

**For PyMOL:**
```bash
pip install pymol-open-source
# or for licensed version:
conda install -c schrodinger pymol
```

### 5. Empty Interaction Sets

**Problem:** `complex.interaction_sets` is empty after analysis.

**Causes:**
- Ligand too far from protein (>6Å typically)
- Ligand not in binding pocket
- Water molecules interfering

**Solution:**
```python
# Check ligand-protein distances
for lig in complex.ligands:
    center = lig.center
    print(f"Ligand center: {center}")
    # Verify center coordinates are within protein bounds
```

### 6. Memory Issues with Large PDB Files

**Problem:** Out of memory with large structures.

**Solutions:**
- Process one ligand at a time
- Remove water molecules before analysis
- Use smaller structure subset

```python
# Process ligands sequentially
for lig in ligands:
    complex_single = PDBComplex()
    complex_single.load_pdb(pdb_file)
    complex_single.characterize_complex(lig)
    complex_single.analyze()
    # Process results...
    del complex_single
```

### 7. Incorrect Molecular Weight

**Problem:** Ligand MW appears wrong (e.g., shows as 0).

**Solution:**
- OpenBabel may not parse ligand correctly
- Check for missing hydrogen atoms
- Verify ligand is a complete molecule

```python
# Alternative: Calculate MW from SMILES if available
from rdkit import Chem
from rdkit.Chem import Descriptors

# If you have SMILES
mol = Chem.MolFromSmiles(smiles)
mw = Descriptors.MolWt(mol)
```

### 8. BindingSiteReport Parsing

**Problem:** Report content doesn't match expected format.

**Solution:**
```python
# Inspect raw report output
from plip.exchange.report import BindingSiteReport

report = BindingSiteReport(interactions)
txt = report.generate_txt()
for line in txt:
    print(repr(line))  # Show raw content
```

## Performance Tips

1. **Batch Processing:** Process multiple PDB files in parallel using multiprocessing
2. **Caching:** Save interaction sets to avoid re-analysis
3. **Selective Analysis:** Only characterize ligands of interest

## Getting Help

- PLIP Documentation: https://plip-tool.biotec.tu-dresden.de/
- PLIP GitHub: https://github.com/pharmai/plip
- PyMOL Wiki: https://pymolwiki.org/