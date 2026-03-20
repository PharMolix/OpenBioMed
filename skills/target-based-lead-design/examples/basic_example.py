"""
Target-Based Lead Design - Complete Example

This script demonstrates the full workflow for generating diverse lead compounds
for a specific protein target using structure-based drug design.
"""

import os
import sys
import pickle
import argparse
import re
import json
import urllib.request
import subprocess
from datetime import datetime
from pathlib import Path

os.environ["dataset_name"] = "BBBP"
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
torch.set_printoptions(precision=4, sci_mode=False)

from open_biomed.tools.tool_registry import TOOLS
from open_biomed.tools.tool_misc import ComplexInteractionAnalysis
from open_biomed.tools.visualization_tools import MoleculeVisualizer
from open_biomed.core.pipeline import InferencePipeline, EnsemblePipeline
from open_biomed.data import Molecule, Protein, Pocket, Text
from pytorch_lightning import seed_everything

# Common ions and cofactors to exclude from ligand selection
EXCLUDED_LIGANDS = {
    # Ions
    'NA', 'CL', 'MG', 'CA', 'ZN', 'K', 'FE', 'MN', 'CU', 'NI', 'CO',
    # Water
    'HOH',
    # Common cofactors
    'NAD', 'NAP', 'NADP', 'FAD', 'FMN', 'ATP', 'ADP', 'AMP', 'GTP', 'GDP',
    'HEM', 'HEME', 'HEC', 'COB', 'B12', 'PLP', 'SAM', 'SAH', 'THM',
    # Lipids/detergents
    'PO4', 'SO4', 'POE', 'PEG', 'TRS', 'EDO', 'DMS', 'ACT', 'FOR',
    # Small molecules
    'GOL', 'PG4', 'P6G', '1PE', 'PGE', 'EOH', 'IPA', 'MOH',
}

# Minimum molecular weight for valid ligand (Daltons)
MIN_LIGAND_MW = 150


def parse_criteria(criteria_str: str) -> dict:
    """Parse criteria from string like 'docking:-10,qed:0.4,lipinski:4'"""
    criteria = {
        'docking_threshold': -10.0,
        'qed_min': 0.4,
        'lipinski_min': 4,
        'side_effects_max': 18,
        'similarity_max': 0.7,
    }
    if criteria_str:
        for item in criteria_str.split(','):
            key, val = item.split(':')
            key = key.strip().lower().replace('-', '_')
            if 'docking' in key:
                criteria['docking_threshold'] = float(val)
            elif 'qed' in key:
                criteria['qed_min'] = float(val)
            elif 'lipinski' in key:
                criteria['lipinski_min'] = int(val)
            elif 'side' in key:
                criteria['side_effects_max'] = int(val)
            elif 'similarity' in key:
                criteria['similarity_max'] = float(val)
    return criteria


def phase1_target_identification(target: str) -> tuple:
    """Phase 1: Get protein structure from PDB ID or disease name."""
    print("=" * 60)
    print("PHASE 1: TARGET IDENTIFICATION")
    print("=" * 60)
    print()

    pdb_tool = TOOLS["protein_pdb_request"]

    # Check if target is a PDB ID (4 characters)
    if len(target) == 4 and target[0].isdigit():
        pdb_id = target.upper()
        print(f"[1.1] Downloading PDB structure: {pdb_id}")
        pdb_file, _ = pdb_tool.run(accession=pdb_id, mode="file_only")
        pdb_file = pdb_file[0]
    else:
        # Treat as disease/target name - search web
        print(f"[1.1] Searching for target: {target}")
        web_search = TOOLS["web_search"]
        results, _ = web_search.run(query=f"{target} protein PDB structure inhibitor")

        # Extract PDB IDs from results
        import re
        pdb_pattern = r'\b[1-9][A-Z0-9]{3}\b'
        pdb_ids = list(set(re.findall(pdb_pattern, str(results))))[:3]

        if not pdb_ids:
            raise ValueError(f"No PDB structures found for target: {target}")

        pdb_id = pdb_ids[0]
        print(f"    Found PDB ID: {pdb_id}")
        pdb_file, _ = pdb_tool.run(accession=pdb_id, mode="file_only")
        pdb_file = pdb_file[0]

    print(f"    Saved to: {pdb_file}")
    return pdb_file, pdb_id


def phase2_structure_preparation(pdb_file: str) -> tuple:
    """Phase 2: Extract protein, ligand, and define binding pocket."""
    print()
    print("=" * 60)
    print("PHASE 2: STRUCTURE PREPARATION")
    print("=" * 60)
    print()

    print("[2.1] Extracting protein chains and ligands...")
    extract_tool = TOOLS["extract_molecules_from_pdb_file"]
    results, metadata = extract_tool.run(pdb_file=pdb_file)

    # Parse results
    extracted = results[0]
    proteins = [(r[1], r[2]) for r in extracted if r[0] == "protein"]
    ligands = [(r[1], r[2]) for r in extracted if r[0] == "molecule"]

    if not proteins:
        raise ValueError("No protein chains found in PDB file")
    if not ligands:
        raise ValueError("No ligand molecules found - cannot define pocket")

    protein = proteins[0][1]
    ligand = ligands[0][1]

    print(f"    Protein: {len(protein.sequence)} residues")
    print(f"    Ligand SMILES: {ligand.smiles}")

    print()
    print("[2.2] Defining binding pocket...")
    pocket = Pocket.from_protein_ref_ligand(protein, ligand, radius=10.0)
    pocket.estimated_num_atoms = ligand.get_num_atoms()
    print(f"    Pocket residues: {len(pocket.orig_indices)}")

    return protein, ligand, pocket


def phase3_molecule_generation(pocket, num_candidates: int, output_dir: Path) -> list:
    """Phase 3: Generate candidate molecules using MolCraft."""
    print()
    print("=" * 60)
    print("PHASE 3: DE NOVO MOLECULE GENERATION")
    print("=" * 60)
    print()

    print(f"[3.1] Generating {num_candidates} candidates using MolCraft...")

    pipeline = InferencePipeline(
        task="structure_based_drug_design",
        model="molcraft",
        model_ckpt="./checkpoints/molcraft/last_updated.ckpt",
        device="cuda:0"
    )

    candidates = []
    for i in range(num_candidates):
        try:
            seed_everything(i * 1000 + 42)
            outputs = pipeline.run(pocket=pocket)
            if outputs and outputs[0] and outputs[0][0]:
                mol = outputs[0][0]
                mol._add_smiles()
                candidates.append(mol)
                if (i + 1) % 10 == 0:
                    print(f"    Generated {len(candidates)} molecules...")
        except Exception as e:
            pass

    print(f"\n    Successfully generated {len(candidates)} candidates")

    # Save candidates to SDF files
    print(f"\n[3.2] Saving candidates to SDF files...")
    for i, mol in enumerate(candidates):
        mol.name = f"candidate_{i+1}"
        sdf_file = mol.save_sdf(str(output_dir / f"candidate_{i+1}.sdf"))

    print(f"    Saved {len(candidates)} SDF files to {output_dir}")
    return candidates


def phase4_docking(protein, candidates: list) -> list:
    """Phase 4: Dock all candidates to target protein."""
    print()
    print("=" * 60)
    print("PHASE 4: DOCKING")
    print("=" * 60)
    print()

    print(f"[4.1] Docking {len(candidates)} candidates...")

    docking_tool = TOOLS["protein_molecule_docking_score"]

    for i, mol in enumerate(candidates):
        try:
            result, _ = docking_tool.run(protein=protein, molecule=mol)
            score = result[0][0] if isinstance(result[0], tuple) else result[0]
            mol.docking_score = float(score)
        except Exception as e:
            mol.docking_score = None

        if (i + 1) % 10 == 0:
            scores = [m.docking_score for m in candidates[:i+1] if m.docking_score]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"    Docked {i+1}/{len(candidates)} (avg: {avg:.2f})")

    valid = sum(1 for m in candidates if m.docking_score is not None)
    print(f"\n    Docked {valid}/{len(candidates)} successfully")
    return candidates


def phase5_property_admet(candidates: list) -> list:
    """Phase 5: Calculate drug-likeness and ADMET properties."""
    print()
    print("=" * 60)
    print("PHASE 5: PROPERTY + ADMET CALCULATION")
    print("=" * 60)
    print()

    print("[5.1] Initializing property tools...")
    qed_tool = TOOLS["molecule_qed"]
    sa_tool = TOOLS["molecule_sa"]
    logp_tool = TOOLS["molecule_logp"]
    lipinski_tool = TOOLS["molecule_lipinski"]

    print("[5.2] Initializing ADMET pipelines...")
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

    print(f"[5.3] Processing {len(candidates)} candidates...")
    for i, mol in enumerate(candidates):
        # Drug-likeness
        qed_result, _ = qed_tool.run(molecule=mol)
        mol.qed = qed_result[0] if isinstance(qed_result, list) else qed_result

        sa_result, _ = sa_tool.run(molecule=mol)
        mol.sa = sa_result[0] if isinstance(sa_result, list) else sa_result

        logp_result, _ = logp_tool.run(molecule=mol)
        mol.logp = logp_result[0] if isinstance(logp_result, list) else logp_result

        lipinski_result, _ = lipinski_tool.run(molecule=mol)
        mol.lipinski = lipinski_result[0] if isinstance(lipinski_result, list) else lipinski_result

        # ADMET
        try:
            bbb_out = admet_pipeline.run(molecule=mol, task="BBBP")
            mol.bbb_prob = float(bbb_out[1][0].strip("[]"))
        except:
            mol.bbb_prob = None

        try:
            sider_out = admet_pipeline.run(molecule=mol, task="SIDER")
            sider_list = eval(sider_out[1][0])
            mol.num_side_effects = sum(1 for s in sider_list if s > 0.5)
        except:
            mol.num_side_effects = None

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(candidates)}")

    return candidates


def phase6_filtering_diversity(candidates: list, criteria: dict, target_leads: int) -> tuple:
    """Phase 6: Apply criteria and select diverse candidates."""
    print()
    print("=" * 60)
    print("PHASE 6: FILTERING & DIVERSITY SELECTION")
    print("=" * 60)
    print()

    print("USER CRITERIA:")
    print(f"  Docking:     <= {criteria['docking_threshold']} kcal/mol")
    print(f"  QED:         >= {criteria['qed_min']}")
    print(f"  Lipinski:    >= {criteria['lipinski_min']} rules obeyed")
    print(f"  Side FX:     <= {criteria['side_effects_max']} categories")
    print(f"  Similarity:  <= {criteria['similarity_max']}")
    print(f"  Target:      {target_leads} diverse leads")
    print()

    # Apply filtering criteria
    print("[6.1] Applying filtering criteria...")
    filtered_indices = []
    fail_stats = {'docking': 0, 'qed': 0, 'lipinski': 0, 'side_effects': 0}

    for i, mol in enumerate(candidates):
        passed = True

        if mol.docking_score is None or mol.docking_score > criteria['docking_threshold']:
            fail_stats['docking'] += 1
            passed = False

        if mol.qed is None or mol.qed < criteria['qed_min']:
            fail_stats['qed'] += 1
            passed = False

        if mol.lipinski is None or mol.lipinski < criteria['lipinski_min']:
            fail_stats['lipinski'] += 1
            passed = False

        if mol.num_side_effects is None or mol.num_side_effects > criteria['side_effects_max']:
            fail_stats['side_effects'] += 1
            passed = False

        if passed:
            filtered_indices.append(i)

    print(f"    Total: {len(candidates)}, Passed: {len(filtered_indices)}")
    print(f"    Failures: docking={fail_stats['docking']}, qed={fail_stats['qed']}, "
          f"lipinski={fail_stats['lipinski']}, side_fx={fail_stats['side_effects']}")

    # Build similarity matrix for filtered candidates
    print()
    print("[6.2] Computing similarity matrix...")
    similarity_tool = TOOLS["molecule_similarity"]
    n = len(filtered_indices)
    sim_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        sim_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            result, _ = similarity_tool.run(
                molecule_1=candidates[filtered_indices[i]],
                molecule_2=candidates[filtered_indices[j]])
            sim = result[0]
            sim_matrix[i][j] = sim_matrix[j][i] = sim

    # Greedy diversity selection
    print("[6.3] Selecting diverse candidates...")
    selected_indices = []

    if filtered_indices:
        selected_indices = [filtered_indices[0]]

        for idx in filtered_indices[1:]:
            is_diverse = True
            for sel_idx in selected_indices:
                i = filtered_indices.index(idx)
                j = filtered_indices.index(sel_idx)
                if sim_matrix[i][j] > criteria['similarity_max']:
                    is_diverse = False
                    break
            if is_diverse:
                selected_indices.append(idx)

    print(f"    Selected {len(selected_indices)} diverse candidates")

    # Regeneration check
    print()
    print("-" * 60)
    print("REGENERATION CHECK")
    print("-" * 60)

    if len(selected_indices) >= target_leads:
        print(f"SUCCESS: {len(selected_indices)} leads selected (target: {target_leads})")
        regeneration_needed = False
    elif len(selected_indices) > 0:
        deficit = target_leads - len(selected_indices)
        print(f"PARTIAL SUCCESS: {len(selected_indices)} leads (target: {target_leads})")
        print(f"OPTIONS:")
        print(f"  1. Generate {deficit * 2}+ more candidates")
        print(f"  2. Relax criteria")
        print(f"  3. Accept current {len(selected_indices)} leads")
        regeneration_needed = True
    else:
        print("NO CANDIDATES PASSED CRITERIA")
        print("OPTIONS:")
        print("  1. Relax criteria significantly")
        print("  2. Generate more candidates")
        regeneration_needed = True

    return selected_indices, sim_matrix, regeneration_needed


def phase7_plip_analysis(protein, candidates: list, selected_indices: list, output_dir: Path) -> list:
    """Phase 7: PLIP interaction analysis for selected molecules only."""
    print()
    print("=" * 60)
    print("PHASE 7: PLIP INTERACTION ANALYSIS")
    print("=" * 60)
    print()

    if not selected_indices:
        print("    No selected molecules to analyze")
        return candidates

    print(f"[7.1] Analyzing protein-ligand interactions for {len(selected_indices)} selected leads...")

    plip_tool = ComplexInteractionAnalysis()

    for rank, idx in enumerate(selected_indices):
        mol = candidates[idx]
        try:
            report, _ = plip_tool.run(molecule=mol, protein=protein)
            mol.interaction_report = report[0] if report else None

            # Save interaction report
            report_file = output_dir / f"interaction_report_{rank+1}.txt"
            with open(report_file, 'w') as f:
                f.write(f"Interaction Report for Candidate {idx+1}\n")
                f.write(f"SMILES: {mol.smiles}\n")
                f.write("=" * 60 + "\n")
                f.write(report[0] if report else "No interactions detected")

            print(f"    [{rank+1}/{len(selected_indices)}] Candidate {idx+1}: Report saved")
        except Exception as e:
            mol.interaction_report = None
            print(f"    [{rank+1}/{len(selected_indices)}] Candidate {idx+1}: Error - {str(e)[:50]}")

    return candidates


def phase8_visualization(protein, candidates: list, selected_indices: list, output_dir: Path) -> list:
    """Phase 8: Visualization for selected molecules only."""
    print()
    print("=" * 60)
    print("PHASE 8: VISUALIZATION")
    print("=" * 60)
    print()

    if not selected_indices:
        print("    No selected molecules to visualize")
        return candidates

    # 8.1: 2D molecule visualization
    print(f"[8.1] Generating 2D molecule visualizations...")
    mol_vis = MoleculeVisualizer()

    for rank, idx in enumerate(selected_indices):
        mol = candidates[idx]
        try:
            img_file, _ = mol_vis.run(
                molecule=mol,
                config='2D',
                img_file=str(output_dir / f"mol_2d_lead_{rank+1}.png")
            )
            print(f"    [{rank+1}/{len(selected_indices)}] 2D: {img_file[0]}")
        except Exception as e:
            print(f"    [{rank+1}/{len(selected_indices)}] 2D Error: {str(e)[:50]}")

    # 8.2: 3D rotating complex visualization (requires PyMOL)
    print()
    print(f"[8.2] Generating 3D rotating complex visualizations (requires PyMOL)...")

    work_dir = Path(__file__).parent.parent.parent

    for rank, idx in enumerate(selected_indices):
        mol = candidates[idx]
        try:
            # Save molecule and protein to files for visualization
            sdf_file = mol.save_sdf(str(output_dir / f"vis_mol_{rank+1}.sdf"))
            pdb_file = protein.save_pdb(str(output_dir / f"vis_protein_{rank+1}.pdb"))
            gif_file = str(output_dir / f"complex_rotating_lead_{rank+1}.gif")

            # Call visualization_tools.py directly to avoid PyMOL import issues
            cmd = [
                'python3',
                str(work_dir / 'open_biomed/tools/visualization_tools.py'),
                '--task', 'visualize_complex',
                '--molecule', sdf_file,
                '--protein', pdb_file,
                '--rotate',
                '--output_file', gif_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if os.path.exists(gif_file):
                print(f"    [{rank+1}/{len(selected_indices)}] 3D: {gif_file}")
            else:
                print(f"    [{rank+1}/{len(selected_indices)}] 3D Error: {result.stderr[:50] if result.stderr else 'File not created'}")

        except Exception as e:
            print(f"    [{rank+1}/{len(selected_indices)}] 3D Error: {str(e)[:50]}")

    return candidates


def generate_report(candidates, selected_indices, pdb_id, output_dir) -> str:
    """Generate comprehensive markdown report."""
    report = []
    report.append(f"# Target-Based Lead Design Report")
    report.append(f"")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Target:** {pdb_id}")
    report.append(f"")
    report.append(f"## Summary")
    report.append(f"")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Total candidates | {len(candidates)} |")
    report.append(f"| Selected leads | {len(selected_indices)} |")
    report.append(f"")
    report.append(f"## Selected Lead Compounds")
    report.append(f"")
    report.append(f"| # | SMILES | Docking | QED | SA | Lipinski | BBB | Side FX |")
    report.append(f"|---|--------|---------|-----|----|----------|----|---------|")

    for rank, idx in enumerate(selected_indices[:20]):
        mol = candidates[idx]
        smiles = mol.smiles[:40] + "..." if len(mol.smiles) > 40 else mol.smiles
        dock = f"{mol.docking_score:.2f}" if mol.docking_score else "N/A"
        qed = f"{mol.qed:.2f}" if mol.qed else "N/A"
        sa = f"{mol.sa:.1f}" if mol.sa else "N/A"
        lip = mol.lipinski if mol.lipinski is not None else "N/A"
        bbb = f"{mol.bbb_prob:.2f}" if mol.bbb_prob else "N/A"
        se = mol.num_side_effects if mol.num_side_effects is not None else "N/A"
        report.append(f"| {rank+1} | `{smiles}` | {dock} | {qed} | {sa} | {lip} | {bbb} | {se} |")

    report.append(f"")
    report.append(f"## Interpretation Guide")
    report.append(f"")
    report.append(f"- **Docking**: < -10 = Excellent, -10 to -7 = Good, > -7 = Moderate")
    report.append(f"- **QED**: > 0.7 = Excellent, 0.5-0.7 = Good, < 0.5 = Poor")
    report.append(f"- **Lipinski**: 4 = No violations, 3 = 1 violation, etc.")
    report.append(f"- **BBB**: > 0.5 = Likely crosses blood-brain barrier")
    report.append(f"- **Side FX**: Lower = fewer predicted side effects")
    report.append(f"")
    report.append(f"## Generated Files")
    report.append(f"")
    report.append(f"- `candidate_*.sdf`: SDF files for all candidates")
    report.append(f"- `interaction_report_*.txt`: PLIP interaction analysis for leads")
    report.append(f"- `mol_2d_lead_*.png`: 2D molecule structures")
    report.append(f"- `complex_rotating_lead_*.gif`: 3D rotating complex visualizations")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Target-Based Lead Design")
    parser.add_argument("--target", type=str, default="4xli", help="PDB ID or disease name")
    parser.add_argument("--num-candidates", type=int, default=40, help="Initial candidates to generate")
    parser.add_argument("--target-leads", type=int, default=20, help="Desired number of final leads")
    parser.add_argument("--criteria", type=str, default="", help="Criteria string, e.g. 'docking:-10,qed:0.4,lipinski:4'")
    parser.add_argument("--output-dir", type=str, default="./outputs/lead_design", help="Output directory")
    args = parser.parse_args()

    # Parse criteria
    criteria = parse_criteria(args.criteria)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run workflow
    pdb_file, pdb_id = phase1_target_identification(args.target)
    protein, ligand, pocket = phase2_structure_preparation(pdb_file)
    candidates = phase3_molecule_generation(pocket, args.num_candidates, output_dir)
    candidates = phase4_docking(protein, candidates)
    candidates = phase5_property_admet(candidates)
    selected_indices, sim_matrix, regen_needed = phase6_filtering_diversity(
        candidates, criteria, args.target_leads)

    # Phase 7 & 8: Only for selected molecules
    if selected_indices:
        candidates = phase7_plip_analysis(protein, candidates, selected_indices, output_dir)
        candidates = phase8_visualization(protein, candidates, selected_indices, output_dir)

    # Generate report
    report = generate_report(candidates, selected_indices, pdb_id, output_dir)
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save results
    results_path = output_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'candidates': candidates,
            'selected_indices': selected_indices,
            'criteria': criteria,
            'pdb_id': pdb_id
        }, f)
    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
