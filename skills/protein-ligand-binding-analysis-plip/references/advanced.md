# Advanced Usage

## Batch Processing Multiple PDB Files

```python
import os
from pathlib import Path
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport

def batch_analyze(pdb_dir, output_dir, min_mw=150):
    """Analyze all PDB files in a directory."""
    results = {}

    for pdb_file in Path(pdb_dir).glob("*.pdb"):
        try:
            complex = PDBComplex()
            complex.load_pdb(str(pdb_file))

            ligands = [lig for lig in complex.ligands if lig.mol.molwt > min_mw]
            for lig in ligands:
                complex.characterize_complex(lig)
            complex.analyze()

            results[pdb_file.name] = {
                "ligands": len(ligands),
                "interactions": len(complex.interaction_sets)
            }
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            results[pdb_file.name] = {"error": str(e)}

    return results
```

## Comparing Interactions Across Structures

```python
def compare_binding_modes(interaction_sets_list):
    """Compare interaction profiles across multiple structures."""
    comparison = {}

    for i, interactions in enumerate(interaction_sets_list):
        profile = {
            "hbonds": len(interactions.all_hbonds_ldon) + len(interactions.all_hbonds_pdon),
            "hydrophobic": len(interactions.hydrophobic_contacts),
            "water_bridges": len(interactions.water_bridges),
        }
        comparison[f"structure_{i}"] = profile

    return comparison
```

## Custom Interaction Filtering

```python
def filter_by_interaction_type(interactions, interaction_type="hbonds"):
    """Extract specific interaction types for detailed analysis."""
    if interaction_type == "hbonds":
        return list(interactions.all_hbonds_ldon) + list(interactions.all_hbonds_pdon)
    elif interaction_type == "hydrophobic":
        return list(interactions.hydrophobic_contacts)
    elif interaction_type == "water_bridges":
        return list(interactions.water_bridges)
    elif interaction_type == "pistacking":
        return list(interactions.pistacking)
    elif interaction_type == "saltbridges":
        return list(interactions.saltbridges)
    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}")
```

## Integration with OpenBioMed Workflow

```python
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.data import Protein

def analyze_with_openbiomed(pdb_path):
    """Use PLIP analysis within OpenBioMed workflow."""
    # Load protein with OpenBioMed
    protein = Protein.from_pdb_file(pdb_path)

    # Run PLIP analysis
    from plip.structure.preparation import PDBComplex
    complex = PDBComplex()
    complex.load_pdb(pdb_path)

    for lig in complex.ligands:
        if lig.mol.molwt > 150:
            complex.characterize_complex(lig)
    complex.analyze()

    return complex.interaction_sets
```

## Custom Visualization Settings

```python
from plip.basic import config

def configure_visualization(
    background="white",
    cartoon=True,
    sticks=True,
    hide_water=True,
    measure=True,
    show_surface=False
):
    """Configure PyMOL visualization settings."""
    config.PICS = True
    config.BACKGROUND = background
    config.CARTOON = cartoon
    config.STICKS = sticks
    config.HIDE_WATER = hide_water
    config.MEASURE = measure
    config.SURFACE = show_surface
```

## Extracting Residue-Level Information

```python
def get_interacting_residues(interactions):
    """Get unique residues involved in interactions."""
    residues = set()

    # From hydrogen bonds
    for hb in interactions.all_hbonds_ldon + interactions.all_hbonds_pdon:
        residues.add((hb.resnr, hb.restype))

    # From hydrophobic contacts
    for hc in interactions.hydrophobic_contacts:
        residues.add((hc.resnr, hc.restype))

    return sorted(residues, key=lambda x: x[0])
```

## Output Formats

### JSON Export

```python
import json

def export_to_json(interaction_sets, output_path):
    """Export interactions to JSON format."""
    data = {}
    for key, interactions in interaction_sets.items():
        data[key] = {
            "hbonds": [
                {"residue": hb.resnr, "type": hb.type}
                for hb in interactions.all_hbonds_ldon + interactions.all_hbonds_pdon
            ],
            "hydrophobic": [
                {"residue": hc.resnr}
                for hc in interactions.hydrophobic_contacts
            ],
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
```

### CSV Summary

```python
import pandas as pd

def export_summary_csv(results, output_path):
    """Export interaction summary to CSV."""
    rows = []
    for ligand_id, stats in results.items():
        rows.append({
            "ligand": ligand_id,
            "hbonds": stats.get("hbonds", 0),
            "hydrophobic": stats.get("hydrophobic", 0),
            "water_bridges": stats.get("water_bridges", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
```