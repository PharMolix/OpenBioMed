---
name: protein-ligand-binding-analysis-plip
description: >
  Analyze protein-ligand interactions in PDB structures using PLIP (Protein-Ligand Interaction Profiler).
  Use this skill when:
  (1) Analyzing binding interactions from a PDB structure file,
  (2) Identifying hydrogen bonds, hydrophobic contacts, π-stacking, salt bridges, and water bridges,
  (3) Generating 3D visualizations of protein-ligand complexes,
  (4) Creating interaction summary reports for drug discovery.
license: MIT
category: binding-affinity
tags: [protein-ligand, interaction-analysis, plip, visualization, docking]
---

# Protein-Ligand Binding Analysis with PLIP

Analyze protein-ligand interactions in PDB structures, generate comprehensive interaction reports, and create 3D visualizations.

## When to Use

- Analyzing binding modes from crystal structures or docking results
- Identifying key interactions driving binding affinity
- Comparing ligand binding patterns across multiple structures
- Generating publication-ready interaction visualizations

## Workflow

### Step 1: Load PDB and Identify Ligands

```python
from plip.structure.preparation import PDBComplex

complex = PDBComplex()
complex.load_pdb(pdb_file)

# Filter ligands by molecular weight (exclude ions/cofactors, MW > 150 Da)
ligands = []
for lig in complex.ligands:
    if lig.mol.molwt > 150:  # OpenBabel molecule object
        ligands.append(lig)
        complex.characterize_complex(lig)
```

### Step 2: Analyze Interactions

```python
from plip.exchange.report import BindingSiteReport

complex.analyze()

for key, interactions in complex.interaction_sets.items():
    report = BindingSiteReport(interactions)
    report_lines = report.generate_txt()
    # Parse interaction data from report_lines
```

### Step 3: Generate Visualizations

```python
from plip.basic.remote import VisualizerData
from plip.visualization.visualize import visualize_in_pymol
from plip.basic import config

config.PICS = True
config.OUTPATH = output_dir
config.BACKGROUND = "white"
config.CARTOON = True

for key in complex.interaction_sets:
    data = VisualizerData(complex, key)
    visualize_in_pymol(data)
```

## Expected Outputs

| Output | Description |
|--------|-------------|
| Interaction Report | Markdown summary of all interaction types per ligand |
| Visualization Images | PNG files showing 3D interaction diagrams |
| Summary Statistics | Counts of H-bonds, hydrophobic, π-stacking, etc. |

### Interaction Types Reported

| Type | Description |
|------|-------------|
| Hydrogen bonds | H-bonds with ligand/protein as donor |
| Hydrophobic contacts | Non-polar interactions |
| Water bridges | Water-mediated interactions |
| π-stacking | Aromatic ring interactions |
| Salt bridges | Ionic interactions |
| Halogen bonds | Halogen-mediated contacts |
| Metal complexes | Metal coordination |

## Score Interpretation

- **More H-bonds**: Generally indicates stronger, more specific binding
- **Hydrophobic contacts**: Contribute to binding entropy
- **Water bridges**: Can enhance or reduce binding affinity
- **π-stacking**: Important for aromatic ligand recognition

## Error Handling

| Error | Solution |
|-------|----------|
| No ligands found | Check PDB file format; ligand may need HETATM records |
| PLIP characterization fails | Ligand may be malformed; try alternative PDB source |
| PyMOL visualization fails | Ensure PyMOL is installed and in PATH |
| Low MW ligands filtered | Adjust MW threshold if small molecules are targets |

## Dependencies

```bash
pip install plip pymol-open-source
```

## Example Usage

```python
# Analyze EGFR-erlotinib complex (1M17)
python examples/basic_example.py --pdb tmp/pdb_1m17.pdb --output ./results/
```

See `examples/basic_example.py` for complete implementation and `references/advanced.md` for batch processing workflows.