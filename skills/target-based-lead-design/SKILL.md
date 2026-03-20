---
name: target-based-lead-design
description: >
  Generate diverse lead compounds for a specific protein target using structure-based
  drug design with MolCraft. Use this skill when:
  (1) Designing drug candidates for a known protein target (PDB ID or disease name),
  (2) Generating structurally diverse molecules with optimized binding affinity,
  (3) Filtering candidates based on user-defined criteria (docking, ADMET, drug-likeness),
  (4) Iteratively refining leads through regeneration when criteria are not met.
license: MIT
category: drug-discovery
tags: [lead-generation, structure-based-design, diversity, molcraft, docking, admet]
---

# Target-Based Lead Design

Generate diverse, drug-like lead compounds targeting a specific protein using AI-powered structure-based drug design.

## When to Use

- User provides a PDB ID or disease name and wants drug candidates
- User wants to design molecules for a specific protein target
- User needs diverse leads with user-defined property criteria
- User wants iterative refinement with regeneration loop

## Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | str | Yes | PDB ID (e.g., "4xli") or disease name |
| `num_candidates` | int | No | Initial candidates to generate (default: 40) |
| `target_leads` | int | No | Desired number of final leads (default: 20) |

### User Criteria (Filtering Thresholds)

| Criterion | Default | Description |
|-----------|---------|-------------|
| `docking_threshold` | -10.0 | Maximum docking score (kcal/mol), more negative = better |
| `qed_min` | 0.4 | Minimum QED score (0-1), higher = more drug-like |
| `lipinski_min` | 4 | Minimum Lipinski rules obeyed (0-4), 4 = no violations |
| `side_effects_max` | 18 | Maximum SIDER side effect categories predicted |
| `similarity_max` | 0.7 | Maximum Tanimoto similarity between selected leads |

## Workflow

```
Phase 1: Target Identification
    └── Path A: PDB ID provided → Download structure directly
    └── Path B: Disease/target name provided → Agent-based discovery:
           ├── Agent searches web for PDB structures
           ├── Agent examines each PDB's ligands
           ├── Agent searches literature to validate ligand is a true binder
           │      └── Fallback (if 3 search attempts fail):
           │             └── Judge by molecular weight:
           │                    • MW ≥ 150 Da → Likely drug-like binder (accept)
           │                    • MW 100-150 Da → Fragment (accept with caution)
           │                    • MW < 100 Da → Likely solvent/ion (exclude)
           ├── Agent ranks by resolution, returns best PDB ID
           └── If no valid PDB found → Ask user for PDB ID

Phase 2: Structure Preparation
    └── Extract protein chains and ligands
    └── Define binding pocket (from reference ligand)

Phase 3: De Novo Generation
    └── Generate candidates using MolCraft
    └── Save candidates to SDF files

Phase 4: Docking
    └── Dock all candidates (AutoDock Vina)

Phase 5: Property + ADMET Calculation
    └── Drug-likeness: QED, SA, LogP, Lipinski
    └── ADMET: BBB penetration, Side effects (SIDER)

Phase 6: Filtering & Diversity Selection
    └── Apply user criteria → Filter candidates
    └── Greedy diversity selection (Tanimoto)
    └── Regeneration check → Iterate if needed

Phase 7: PLIP Interaction Analysis (selected molecules only)
    └── Analyze protein-ligand interactions for selected leads
    └── Report hydrophobic contacts, H-bonds, π-stacking, salt bridges

Phase 8: Visualization (selected molecules only)
    └── 2D molecule structures (RDKit)
    └── 3D rotating complex GIF (PyMOL, requires installation)
```

## Core Implementation

### Phase 1-2: Target Retrieval & Pocket Definition

```python
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.data import Pocket

# Download PDB structure
pdb_tool = TOOLS["protein_pdb_request"]
pdb_file, _ = pdb_tool.run(accession="4xli", mode="file_only")

# Extract protein and ligand
extract_tool = TOOLS["extract_molecules_from_pdb_file"]
results, _ = extract_tool.run(pdb_file=pdb_file[0])
# results[0] contains list of (type, chain_id, entity) tuples

protein = [r[2] for r in results[0] if r[0] == "protein"][0]
ligand = [r[2] for r in results[0] if r[0] == "molecule"][0]

# Define pocket from reference ligand
pocket = Pocket.from_protein_ref_ligand(protein, ligand, radius=10.0)
pocket.estimated_num_atoms = ligand.get_num_atoms()
```

### Phase 3: Molecule Generation

```python
from open_biomed.core.pipeline import InferencePipeline
from pytorch_lightning import seed_everything

pipeline = InferencePipeline(
    task="structure_based_drug_design",
    model="molcraft",
    model_ckpt="./checkpoints/molcraft/last_updated.ckpt",
    device="cuda:0"
)

candidates = []
for i in range(num_candidates):
    seed_everything(i * 1000 + 42)
    outputs = pipeline.run(pocket=pocket)
    if outputs and outputs[0] and outputs[0][0]:
        mol = outputs[0][0]
        mol._add_smiles()
        candidates.append(mol)
```

### Phase 4: Docking

```python
docking_tool = TOOLS["protein_molecule_docking_score"]

for mol in candidates:
    result, _ = docking_tool.run(protein=protein, molecule=mol)
    score = result[0][0]  # (score, docked_molecule) tuple
    mol.docking_score = score
```

### Phase 5: Property & ADMET

```python
from open_biomed.core.pipeline import InferencePipeline, EnsemblePipeline

# Drug-likeness tools
qed_tool = TOOLS["molecule_qed"]
sa_tool = TOOLS["molecule_sa"]
logp_tool = TOOLS["molecule_logp"]
lipinski_tool = TOOLS["molecule_lipinski"]

# ADMET pipeline
pipelines = {
    "BBBP": InferencePipeline(
        task="molecule_property_prediction", model="graphmvp",
        model_ckpt="./checkpoints/server/graphmvp-BBBP.ckpt",
        additional_config="./configs/dataset/bbbp.yaml", device="cuda:0"),
    "SIDER": InferencePipeline(
        task="molecule_property_prediction", model="graphmvp",
        model_ckpt="./checkpoints/server/graphmvp-SIDER.ckpt",
        additional_config="./configs/dataset/sider.yaml", device="cuda:0"),
}
admet_pipeline = EnsemblePipeline(pipelines)

for mol in candidates:
    # Drug-likeness
    qed, _ = qed_tool.run(molecule=mol)
    sa, _ = sa_tool.run(molecule=mol)
    logp, _ = logp_tool.run(molecule=mol)
    lipinski, _ = lipinski_tool.run(molecule=mol)

    mol.qed = qed[0]
    mol.sa = sa[0]
    mol.logp = logp[0]
    mol.lipinski = lipinski[0]  # Rules obeyed (0-4)

    # ADMET
    bbb_out = admet_pipeline.run(molecule=mol, task="BBBP")
    mol.bbb_prob = float(bbb_out[1][0].strip("[]"))

    sider_out = admet_pipeline.run(molecule=mol, task="SIDER")
    sider_list = eval(sider_out[1][0])
    mol.num_side_effects = sum(1 for s in sider_list if s > 0.5)
```

### Phase 6: Filtering & Diversity

```python
similarity_tool = TOOLS["molecule_similarity"]

# Apply user criteria
filtered = [i for i, mol in enumerate(candidates) if
    mol.docking_score <= docking_threshold and
    mol.qed >= qed_min and
    mol.lipinski >= lipinski_min and
    mol.num_side_effects <= side_effects_max]

# Build similarity matrix
n = len(filtered)
sim_matrix = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        sim, _ = similarity_tool.run(
            molecule_1=candidates[filtered[i]],
            molecule_2=candidates[filtered[j]])
        sim_matrix[i][j] = sim_matrix[j][i] = sim[0]

# Greedy diversity selection
selected = [filtered[0]]
for idx in filtered[1:]:
    is_diverse = all(
        similarity_matrix[idx][s] <= similarity_max
        for s in selected)
    if is_diverse:
        selected.append(idx)
```

### Regeneration Loop

```python
while len(selected) < target_leads and attempts < max_attempts:
    print(f"Only {len(selected)} leads, need {target_leads}")
    print("Options: 1) Generate more, 2) Relax criteria, 3) Accept")
    # User chooses action
    if user_choice == "generate":
        new_candidates = generate_more(num_additional)
        candidates.extend(new_candidates)
        # Re-run from Phase 4
    elif user_choice == "relax":
        qed_min = max(0.3, qed_min - 0.1)
        side_effects_max += 3
        # Re-filter
```

### Phase 7: PLIP Interaction Analysis (Selected Leads Only)

```python
from open_biomed.tools.tool_misc import ComplexInteractionAnalysis

plip_tool = ComplexInteractionAnalysis()

for idx in selected:
    mol = candidates[idx]
    report, _ = plip_tool.run(molecule=mol, protein=protein)
    # Report contains: hydrophobic interactions, H-bonds,
    # π-stacking, salt bridges, water bridges, etc.
    mol.interaction_report = report[0]
```

### Phase 8: Visualization (Selected Leads Only)

```python
import subprocess
from rdkit import Chem
from plip.structure.preparation import PDBComplex
from plip.basic.remote import VisualizerData
from plip.visualization.visualize import visualize_in_pymol
from plip.basic import config
from open_biomed.tools.visualization_tools import MoleculeVisualizer, ComplexVisualizer
from open_biomed.data import Pocket, Protein

# 2D molecule visualization
mol_vis = MoleculeVisualizer()
for idx in selected:
    mol = candidates[idx]
    img_file, _ = mol_vis.run(molecule=mol, config='2D',
        img_file=f'./outputs/mol_2d_{idx}.png')

# 3D rotating complex visualization (requires PyMOL)
# Full protein view with surface mode
complex_vis = ComplexVisualizer()
for idx in selected:
    mol = candidates[idx]

    # Full protein-ligand complex view
    gif_file = f'./outputs/complex_rotating_{idx}.gif'
    complex_vis.run(
        molecule=mol,
        protein=protein,
        molecule_config='ball_and_stick',
        protein_config='surface',
        img_file=gif_file,
        rotate=True
    )

    # Zoomed view: pocket-ligand complex only
    # Extract pocket around ligand and save as PDB
    pocket = Pocket.from_protein_ref_ligand(protein, mol, radius=10.0)
    pocket_pdb_file = pocket.save_pdb(f'./outputs/pocket_{idx}.pdb')

    # Load pocket PDB as Protein for visualization
    pocket_protein = Protein.from_pdb_file(pocket_pdb_file)

    gif_file_zoomed = f'./outputs/complex_zoomed_{idx}.gif'
    complex_vis.run(
        molecule=mol,
        protein=pocket_protein,
        molecule_config='ball_and_stick',
        protein_config='surface',
        img_file=gif_file_zoomed,
        rotate=True
    )

# PLIP interaction visualization (requires PyMOL and PLIP)
# Shows protein-ligand interactions with annotated H-bonds, hydrophobic contacts, etc.
for idx in selected:
    mol = candidates[idx]

    # Create combined complex PDB file for PLIP
    sdf_file = mol.save_sdf(f'./outputs/mol_{idx}.sdf')
    pdb_file = protein.save_pdb(f'./outputs/protein_{idx}.pdb')

    rdmol = Chem.MolFromMolFile(sdf_file)
    rdprotein = Chem.MolFromPDBFile(pdb_file, sanitize=False)
    rdcomplex = Chem.CombineMols(rdmol, rdprotein)
    complex_pdb_file = f'./outputs/complex_plip_{idx}.pdb'
    Chem.MolToPDBFile(rdcomplex, complex_pdb_file)

    # Run PLIP analysis and visualization
    complex_obj = PDBComplex()
    complex_obj.load_pdb(complex_pdb_file)
    for ligand in complex_obj.ligands:
        complex_obj.characterize_complex(ligand)
    complex_obj.analyze()

    # Generate visualization for each ligand binding site
    for key in complex_obj.interaction_sets:
        data = VisualizerData(complex_obj, key)
        config.PICS = True
        config.OUTPATH = f'./outputs/plip_viz_{idx}'
        config.BACKGROUND = "white"
        config.CARTOON = True
        config.STICKS = True
        config.HIDE_WATER = True
        visualize_in_pymol(data)
```

## Expected Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Lead compounds | List[dict] | SMILES, docking score, properties |
| Diversity report | Table | Pairwise Tanimoto similarities |
| ADMET profile | Table | BBB, side effects per candidate |
| Interaction reports | List[str] | PLIP analysis for selected leads |
| 2D structures | PNG files | Molecule diagrams |
| 3D complexes | GIF files | Rotating protein-ligand visualizations (full view) |
| 3D zoomed complexes | GIF files | Rotating pocket-ligand visualizations (zoomed view) |
| PLIP interactions | PNG files | Protein-ligand interactions with annotated H-bonds, hydrophobic contacts, etc. |
| Summary report | Markdown | Comprehensive lead analysis |

## Output Interpretation

### Docking Score (kcal/mol)
| Score | Assessment |
|-------|------------|
| < -10 | Excellent binding |
| -10 to -7 | Good binding |
| -7 to -5 | Moderate binding |
| > -5 | Weak binding |

### QED (Quantitative Estimate of Drug-likeness)
| Score | Assessment |
|-------|------------|
| > 0.7 | Excellent drug-likeness |
| 0.5 - 0.7 | Good drug-likeness |
| 0.4 - 0.5 | Acceptable |
| < 0.4 | Poor drug-likeness |

### Lipinski Rules Obeyed
| Count | Violations | Assessment |
|-------|------------|------------|
| 4 | 0 | Perfect compliance |
| 3 | 1 | Acceptable |
| 2 | 2 | Marginal |
| < 2 | > 2 | May have issues |

### BBB Penetration Probability
| Probability | Interpretation |
|-------------|----------------|
| > 0.5 | Likely crosses BBB (CNS drug) |
| < 0.5 | Unlikely to cross BBB |

### Side Effects (SIDER categories)
| Count | Risk Level |
|-------|------------|
| 0-10 | Low risk |
| 10-15 | Moderate risk |
| 15-20 | Elevated risk |
| > 20 | High risk |

## Error Handling

| Error | Solution |
|-------|----------|
| PDB not found | Check PDB ID validity or use disease name |
| No ligand in PDB | Use binding site prediction tool |
| MolCraft checkpoint missing | Check `./checkpoints/molcraft/` |
| No candidates pass criteria | Relax criteria or generate more |
| CUDA OOM | Use CPU or reduce batch size |

## Example Usage

```
Input:
  target: "4xli" (ABL2 kinase)
  num_candidates: 40
  target_leads: 20
  criteria:
    docking_threshold: -10
    qed_min: 0.4
    lipinski_min: 4
    side_effects_max: 18
    similarity_max: 0.7

Output:
  6 diverse leads selected
  (Regeneration suggested: generate 28+ more candidates)
```

## See Also

- `examples/basic_example.py` - Complete runnable workflow
- `references/interpretation_guide.md` - Detailed property interpretation
- `references/regeneration_strategies.md` - When and how to regenerate
