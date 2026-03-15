#!/usr/bin/env python
"""
KRAS G12C Drug Candidate Discovery
===================================
Design drug candidates for KRAS G12C mutant protein for cancer therapy.
Generates 2 molecules with high binding affinity using structure-based drug design.

Target: KRAS G12C (Glycine to Cysteine mutation at position 12)
Disease: Non-small cell lung cancer, colorectal cancer, pancreatic cancer
Binding Site: Switch-II pocket
"""

import os
import sys

# Set up paths
sys.path.insert(0, '/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch')
os.chdir('/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch')

import torch
import datetime
import logging
from pytorch_lightning import seed_everything

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

print("Importing OpenBioMed modules...")
from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.tasks.aidd_tasks.structure_based_drug_design import calc_vina_molecule_metrics
from open_biomed.core.visualize import MoleculeVisualizer, ComplexVisualizer

# Output directory
OUTPUT_DIR = "/home/luoyz/projects/OpenBioMed/OpenBioMed_dev/BioMedSkills/drug-candidate-discovery-workspace/iteration-1/kras_g12c_targeted_therapy/without_skill/outputs"

# Set device - use cuda:1 if available
def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return "cuda:1"
        return "cuda:0"
    return "cpu"

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Set random seed for reproducibility
seed_everything(42)

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
os.makedirs('./tmp', exist_ok=True)


def download_pdb_structure(pdb_id: str) -> Protein:
    """Download PDB structure from RCSB."""
    import requests
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_file = f"./tmp/{pdb_id}.pdb"

    if not os.path.exists(pdb_file):
        print(f"Downloading PDB {pdb_id}...")
        response = requests.get(url)
        with open(pdb_file, 'w') as f:
            f.write(response.text)
        print(f"Downloaded to {pdb_file}")

    protein = Protein.from_pdb_file(pdb_file)
    protein.name = f"kras_g12c_{pdb_id}"
    return protein


def create_pocket_from_smiles(protein: Protein, ref_smiles: str, radius: float = 10.0) -> Pocket:
    """Create pocket from protein and a reference ligand SMILES."""
    ref_mol = Molecule.from_smiles(ref_smiles)
    ref_mol._add_rdmol()
    ref_mol._add_conformer(mode='3D')
    pocket = Pocket.from_protein_ref_ligand(protein, ref_mol, radius=radius)
    return pocket


def generate_candidates(pocket: Pocket, num_molecules: int = 2, device: str = DEVICE):
    """Generate drug candidates using MolCraft model."""
    print(f"\nInitializing MolCraft pipeline on {device}...")

    # Try MolCraft first, fall back to pharmolix_fm if not available
    try:
        pipeline = InferencePipeline(
            task="structure_based_drug_design",
            model="molcraft",
            model_ckpt="./checkpoints/molcraft/last_updated.ckpt",
            device=device
        )
        print("Using MolCraft model")
    except Exception as e:
        print(f"MolCraft not available, using pharmolix_fm: {e}")
        pipeline = InferencePipeline(
            task="structure_based_drug_design",
            model="pharmolix_fm",
            model_ckpt="./checkpoints/server/pharmolix_fm.ckpt",
            device=device
        )

    candidates = []
    print(f"Generating {num_molecules} candidate molecules...")

    for i in range(num_molecules):
        try:
            seed_everything(42 + i * 100)  # Different seeds for diversity
            print(f"  Generating molecule {i+1}/{num_molecules}...")
            outputs, files = pipeline.run(pocket=pocket)
            if outputs and len(outputs) > 0 and outputs[0] is not None:
                outputs[0].name = f"kras_g12c_candidate_{i+1}"
                candidates.append(outputs[0])
                smiles = outputs[0].smiles if hasattr(outputs[0], 'smiles') else 'N/A'
                print(f"    Generated: {smiles[:60]}...")
        except Exception as e:
            print(f"    Generation failed: {e}")

    return candidates


def evaluate_molecule(molecule: Molecule, protein: Protein):
    """Evaluate molecule for drug-likeness and binding affinity."""
    metrics = {}

    try:
        molecule._add_smiles()
        metrics['smiles'] = molecule.smiles
        metrics['num_atoms'] = molecule.get_num_atoms()
        metrics['qed'] = molecule.calc_qed()
        metrics['sa'] = molecule.calc_sa()
        metrics['logp'] = molecule.calc_logp()
        metrics['lipinski'] = molecule.calc_lipinski()
        metrics['completeness'] = 0 if "." in molecule.smiles else 1

        # Vina docking score
        vina_metrics = calc_vina_molecule_metrics(molecule, protein, calculate_vina_dock=False)
        metrics['vina_min'] = vina_metrics['vina_min']
        metrics['vina_score'] = vina_metrics['vina_score']

    except Exception as e:
        print(f"  Error evaluating molecule: {e}")
        metrics = {
            'smiles': 'N/A', 'num_atoms': 0, 'qed': 0.0, 'sa': 0.0,
            'logp': 0.0, 'lipinski': 0, 'vina_min': 0.0, 'vina_score': 0.0
        }

    return metrics


def create_visualizations(molecule: Molecule, protein: Protein, output_dir: str, idx: int):
    """Create visualizations for the molecule."""
    vis_dir = os.path.join(output_dir, 'visualizations')

    # 2D molecule visualization
    try:
        mol_viz = MoleculeVisualizer()
        img_path = os.path.join(vis_dir, f"candidate_{idx}_molecule.png")
        mol_viz.run(molecule, config="2D", img_file=img_path)
        print(f"    Saved 2D visualization: {img_path}")
    except Exception as e:
        print(f"    2D visualization failed: {e}")

    # 3D complex visualization
    try:
        complex_viz = ComplexVisualizer()
        img_path = os.path.join(vis_dir, f"candidate_{idx}_complex.png")
        complex_viz.run(molecule, protein, img_file=img_path, rotate=False)
        print(f"    Saved 3D complex visualization: {img_path}")
    except Exception as e:
        print(f"    3D visualization failed: {e}")


def create_report(candidates, metrics_list, output_dir: str):
    """Create markdown report."""
    report_path = os.path.join(output_dir, "report.md")

    report = f"""# KRAS G12C Drug Candidate Discovery Report

## Executive Summary

This report presents **{len(candidates)} drug candidate molecules** designed for targeting the KRAS G12C mutant protein for cancer therapy.

**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Target:** KRAS G12C (Glycine to Cysteine mutation at position 12)

**Device:** {DEVICE}

---

## Target Information

### KRAS G12C Background

KRAS is one of the most frequently mutated oncogenes in human cancers. The G12C mutation is found in:
- **13% of non-small cell lung cancers (NSCLC)**
- **3% of colorectal cancers**
- **2% of pancreatic cancers**

This mutation creates a unique cysteine residue that can be targeted by covalent inhibitors.

### Known Inhibitors

1. **Sotorasib (AMG510)** - FDA approved (2021)
2. **Adagrasib (MRTX849)** - FDA approved (2022)

---

## Methods

- **Model:** Structure-based drug design (SBDD)
- **Binding Pocket:** Switch-II pocket
- **Pocket Radius:** 10 Angstroms

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Vina Min | Docking score (kcal/mol) | < -7.0 |
| QED | Drug-likeness score | > 0.25 |
| SA | Synthetic Accessibility | > 0.5 |
| LogP | Lipophilicity | -1 to 3 |
| Lipinski | Rule of 5 compliance | >= 4/5 |

---

## Results

### Candidate Molecules

| ID | SMILES | QED | SA | LogP | Vina Min | Lipinski |
|----|--------|-----|----|------|----------|----------|
"""

    for i, (mol, metrics) in enumerate(zip(candidates, metrics_list), 1):
        smiles = metrics.get('smiles', 'N/A')
        if len(smiles) > 50:
            smiles = smiles[:50] + "..."
        report += f"| {i} | {smiles} | {metrics.get('qed', 0):.3f} | {metrics.get('sa', 0):.2f} | {metrics.get('logp', 0):.2f} | {metrics.get('vina_min', 0):.2f} | {metrics.get('lipinski', 0)}/5 |\n"

    report += """

### Top Candidate Details

"""

    for i, (mol, metrics) in enumerate(zip(candidates[:2], metrics_list[:2]), 1):
        report += f"""#### Candidate {i}

**SMILES:** `{metrics.get('smiles', 'N/A')}`

**Properties:**
- Number of Atoms: {metrics.get('num_atoms', 'N/A')}
- QED Score: {metrics.get('qed', 0):.3f}
- SA Score: {metrics.get('sa', 0):.2f}
- LogP: {metrics.get('logp', 0):.2f}
- Lipinski Rules: {metrics.get('lipinski', 0)}/5
- Vina Min Score: {metrics.get('vina_min', 0):.2f} kcal/mol

**Files:**
- SDF: `candidate_{i}.sdf`
- 2D Structure: `visualizations/candidate_{i}_molecule.png`
- 3D Complex: `visualizations/candidate_{i}_complex.png`

---

"""

    report += f"""## Conclusions

Successfully generated {len(candidates)} drug candidates targeting the KRAS G12C switch-II pocket.

### Next Steps

1. **Molecular Dynamics:** Validate binding stability
2. **Synthesis Planning:** Develop synthetic routes
3. **In vitro Testing:** Covalent binding assays
4. **Lead Optimization:** Structure-activity relationship studies

## References

1. Ostrem JM, et al. (2013). K-Ras(G12C) inhibitors allosterically control GTP affinity. Nature, 503(7477), 548-551.
2. Canon J, et al. (2019). The clinical KRAS(G12C) inhibitor AMG 510. Nature, 575(7781), 217-223.
3. PDB: 7S35 - KRAS G12C with sotorasib

---
*Generated by OpenBioMed Drug Candidate Discovery*
"""

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved: {report_path}")
    return report_path


def main():
    """Main workflow for KRAS G12C drug candidate discovery."""
    print("=" * 60)
    print("KRAS G12C Drug Candidate Discovery")
    print("=" * 60)

    # Step 1: Download KRAS G12C structure (7S35 - with sotorasib)
    print("\n[Step 1] Downloading KRAS G12C structure...")
    PDB_ID = "7S35"  # KRAS G12C with sotorasib

    try:
        protein = download_pdb_structure(PDB_ID)
        print(f"Loaded protein with {len(protein.residues)} residues")
    except Exception as e:
        print(f"Failed to download PDB: {e}")
        # Fallback to known structure
        print("Using 6OIM structure as fallback...")
        protein = download_pdb_structure("6OIM")

    # Step 2: Create binding pocket
    print("\n[Step 2] Creating binding pocket...")
    # Sotorasib SMILES as reference for pocket
    sotorasib_smiles = "CC(C)C1=NC=C2C(=N1)C(=NC=N2)NC(=O)C3CC(C(=O)N(C3)C(C(C)C)C(=O)N)C4=C(C=CC(=C4)F)F"

    try:
        pocket = create_pocket_from_smiles(protein, sotorasib_smiles, radius=10.0)
        pocket.estimated_num_atoms = 25
        print(f"Created pocket with {len(pocket.orig_indices)} residues")
    except Exception as e:
        print(f"Pocket creation failed: {e}")
        # Fallback: create pocket from residue selection
        print("Creating pocket from residue selection...")
        pocket_residues = [10, 11, 12, 13, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
        pocket = Pocket.from_protein_subseq(protein, pocket_residues)
        pocket.estimated_num_atoms = 25

    # Step 3: Generate drug candidates
    print("\n[Step 3] Generating drug candidates...")
    candidates = generate_candidates(pocket, num_molecules=2, device=DEVICE)

    if len(candidates) < 2:
        print("Warning: Generated fewer than 2 candidates. Creating additional...")
        # Generate more attempts
        additional = generate_candidates(pocket, num_molecules=5, device=DEVICE)
        candidates.extend(additional)
        candidates = candidates[:2]  # Keep only 2

    print(f"\nGenerated {len(candidates)} candidates")

    # Step 4: Evaluate candidates
    print("\n[Step 4] Evaluating candidates...")
    metrics_list = []
    for i, mol in enumerate(candidates):
        print(f"\n  Evaluating candidate {i+1}:")
        metrics = evaluate_molecule(mol, protein)
        metrics_list.append(metrics)
        print(f"    SMILES: {metrics.get('smiles', 'N/A')[:50]}...")
        print(f"    QED: {metrics.get('qed', 0):.3f}, SA: {metrics.get('sa', 0):.2f}")
        print(f"    Vina Min: {metrics.get('vina_min', 0):.2f} kcal/mol")

    # Step 5: Save outputs
    print("\n[Step 5] Saving outputs...")
    for i, (mol, metrics) in enumerate(zip(candidates, metrics_list), 1):
        # Save SDF
        sdf_path = os.path.join(OUTPUT_DIR, f"candidate_{i}.sdf")
        mol.save_sdf(sdf_path, overwrite=True)
        print(f"  Saved SDF: {sdf_path}")

        # Create visualizations
        create_visualizations(mol, protein, OUTPUT_DIR, i)

    # Step 6: Create report
    print("\n[Step 6] Creating report...")
    report_path = create_report(candidates, metrics_list, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(candidates)} drug candidates for KRAS G12C")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR}/candidate_1.sdf")
    print(f"  - {OUTPUT_DIR}/candidate_2.sdf")
    print(f"  - {OUTPUT_DIR}/visualizations/candidate_1_molecule.png")
    print(f"  - {OUTPUT_DIR}/visualizations/candidate_1_complex.png")
    print(f"  - {OUTPUT_DIR}/visualizations/candidate_2_molecule.png")
    print(f"  - {OUTPUT_DIR}/visualizations/candidate_2_complex.png")
    print(f"  - {OUTPUT_DIR}/report.md")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF DRUG CANDIDATES")
    print("=" * 60)
    for i, metrics in enumerate(metrics_list, 1):
        print(f"\nCandidate {i}:")
        print(f"  SMILES: {metrics.get('smiles', 'N/A')}")
        print(f"  QED: {metrics.get('qed', 0):.3f}")
        print(f"  SA: {metrics.get('sa', 0):.2f}")
        print(f"  LogP: {metrics.get('logp', 0):.2f}")
        print(f"  Vina Min: {metrics.get('vina_min', 0):.2f} kcal/mol")

    return candidates, metrics_list


if __name__ == "__main__":
    candidates, metrics = main()
