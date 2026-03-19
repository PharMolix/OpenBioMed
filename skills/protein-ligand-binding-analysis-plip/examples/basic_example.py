#!/usr/bin/env python
"""
Basic example for Protein-Ligand Binding Analysis with PLIP.

Analyzes a PDB file, identifies protein-ligand interactions,
generates a markdown report, and creates 3D visualizations.

Usage:
    python basic_example.py --pdb /path/to/structure.pdb --output ./results/
"""

import os
import argparse
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
from plip.basic.remote import VisualizerData
from plip.visualization.visualize import visualize_in_pymol
from plip.basic import config


def filter_ligands_by_mw(complex, min_mw=150):
    """Filter ligands by molecular weight, excluding ions and cofactors."""
    valid_ligands = []
    for lig in complex.ligands:
        mw = lig.mol.molwt  # OpenBabel molecule object
        if mw > min_mw:
            valid_ligands.append(lig)
    return valid_ligands


def analyze_interactions(complex, ligands):
    """Analyze and characterize all valid ligand complexes."""
    for lig in ligands:
        complex.characterize_complex(lig)
    complex.analyze()
    return complex.interaction_sets


def generate_markdown_report(interaction_sets, output_path):
    """Generate a markdown summary report of all interactions."""
    report_lines = ["# Protein-Ligand Interaction Analysis Report\n"]

    for key, interactions in interaction_sets.items():
        bs_report = BindingSiteReport(interactions)
        txt_report = bs_report.generate_txt()

        report_lines.append(f"## Ligand: {key}\n")
        report_lines.append(f"### Interaction Summary\n")

        # Parse the text report
        for line in txt_report:
            if line.strip():
                report_lines.append(f"- {line}")

        # Add statistics
        stats = {
            "Hydrogen Bonds": len(interactions.all_hbonds_ldon) + len(interactions.all_hbonds_pdon),
            "Hydrophobic Contacts": len(interactions.hydrophobic_contacts),
            "Water Bridges": len(interactions.water_bridges),
            "Pi-Stacking": len(interactions.pistacking),
            "Salt Bridges": len(interactions.saltbridges),
            "Halogen Bonds": len(interactions.halogen_bonds),
            "Metal Complexes": len(interactions.metal_complexes),
        }

        report_lines.append("\n### Statistics\n")
        report_lines.append("| Interaction Type | Count |")
        report_lines.append("|-----------------|-------|")
        for itype, count in stats.items():
            report_lines.append(f"| {itype} | {count} |")
        report_lines.append("\n")

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    return output_path


def generate_visualizations(complex, interaction_sets, output_dir):
    """Generate 3D visualization images using PyMOL."""
    config.PICS = True
    config.OUTPATH = output_dir
    config.BACKGROUND = "white"
    config.CARTOON = True
    config.STICKS = True
    config.HIDE_WATER = True

    image_paths = []
    for key in interaction_sets.keys():
        data = VisualizerData(complex, key)
        visualize_in_pymol(data)
        # Expected output file naming: {pdb_id}_{ligand_id}_{chain}_{position}.png
        image_paths.append(os.path.join(output_dir, f"{key}.png"))

    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Analyze protein-ligand interactions")
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--min-mw", type=float, default=150, help="Minimum ligand MW (Da)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Load PDB and identify ligands
    print(f"Loading PDB file: {args.pdb}")
    complex = PDBComplex()
    complex.load_pdb(args.pdb)

    ligands = filter_ligands_by_mw(complex, min_mw=args.min_mw)
    print(f"Found {len(ligands)} ligand(s) with MW > {args.min_mw} Da")

    if not ligands:
        print("No valid ligands found. Exiting.")
        return

    # Print ligand info
    for lig in ligands:
        print(f"  - {lig.hetid} (MW: {lig.mol.molwt:.2f} Da)")

    # Step 2: Analyze interactions
    print("\nAnalyzing interactions...")
    interaction_sets = analyze_interactions(complex, ligands)

    # Step 3: Generate report
    report_path = os.path.join(args.output, "interaction_report.md")
    generate_markdown_report(interaction_sets, report_path)
    print(f"Report saved to: {report_path}")

    # Step 4: Generate visualizations
    print("\nGenerating visualizations...")
    image_paths = generate_visualizations(complex, interaction_sets, args.output)
    print(f"Visualizations saved to: {args.output}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
