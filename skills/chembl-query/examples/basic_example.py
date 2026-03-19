"""
ChEMBL Query Examples

This script demonstrates all three query types for the ChEMBL query skill.

Run from the OpenBioMed root directory:
    python skills/chembl-query/examples/basic_example.py
"""

import os
import sys

# Add the OpenBioMed root to path if running from skills directory
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from open_biomed.tools.tool_registry import TOOLS


def example_target_search():
    """Find potent EGFR inhibitors."""
    print("=" * 60)
    print("Example 1: Target-Based Compound Search (EGFR)")
    print("=" * 60)

    tool = TOOLS["chembl_query"]

    results, _ = tool.run(
        query_type="target",
        target_name="EGFR",
        standard_type="IC50",
        standard_value_lte=100,  # Sub-100 nM potency
        limit=10
    )

    print(f"\nFound {len(results)} potent EGFR inhibitors (IC50 < 100 nM):\n")
    for r in results:
        name = r.get("molecule_name") or r["molecule_chembl_id"]
        value = r.get("standard_value", "N/A")
        units = r.get("standard_units", "")
        pchembl = r.get("pchembl_value", "N/A")
        print(f"  {name:<25} IC50={value} {units}  pChEMBL={pchembl}")

    return results


def example_molecule_search():
    """Get bioactivity profile for imatinib."""
    print("\n" + "=" * 60)
    print("Example 2: Molecule Bioactivity Profile (Imatinib)")
    print("=" * 60)

    tool = TOOLS["chembl_query"]

    results, _ = tool.run(
        query_type="molecule",
        molecule_name="imatinib",
        limit=20
    )

    print(f"\nFound {len(results)} activity records for imatinib:\n")

    # Group by target
    targets = {}
    for r in results:
        target = r.get("target_name", "Unknown")
        if target not in targets:
            targets[target] = []
        targets[target].append(r)

    for target, activities in list(targets.items())[:5]:
        print(f"  Target: {target}")
        for act in activities[:2]:
            act_type = act.get("standard_type", "N/A")
            value = act.get("standard_value", "N/A")
            units = act.get("standard_units", "")
            print(f"    - {act_type}={value} {units}")
        print()

    return results


def example_smiles_search():
    """Search molecule by SMILES."""
    print("=" * 60)
    print("Example 3: Molecule Search by SMILES (Aspirin)")
    print("=" * 60)

    tool = TOOLS["chembl_query"]

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"

    results, _ = tool.run(
        query_type="molecule",
        smiles=aspirin_smiles,
        limit=10
    )

    if results:
        print(f"\nFound {len(results)} activity records for aspirin (CHEMBL25):\n")
        for r in results[:5]:
            target = r.get("target_name", "Unknown")[:30]
            act_type = r.get("standard_type", "N/A")
            value = r.get("standard_value", "N/A")
            print(f"  {target:<30} {act_type}={value}")
    else:
        print("No results found")

    return results


def example_indication_search():
    """Find approved drugs for diabetes."""
    print("\n" + "=" * 60)
    print("Example 4: Disease Indication Search (Diabetes)")
    print("=" * 60)

    tool = TOOLS["chembl_query"]

    # Find approved diabetes drugs
    results, _ = tool.run(
        query_type="indication",
        disease="diabetes",
        max_phase=4,  # Approved only
        limit=20
    )

    print(f"\nFound {len(results)} approved diabetes drugs:\n")
    for r in results[:10]:
        name = r.get("molecule_name") or r["molecule_chembl_id"]
        indication = r.get("indication", "")
        phase = r.get("phase_description", "")
        print(f"  {name:<20} [{phase}] {indication}")

    return results


def example_uniprot_search():
    """Search target by UniProt ID."""
    print("\n" + "=" * 60)
    print("Example 5: Target Search by UniProt ID (P04637 = TP53)")
    print("=" * 60)

    tool = TOOLS["chembl_query"]

    results, _ = tool.run(
        query_type="target",
        uniprot_id="P04637",  # TP53
        standard_type="IC50",
        limit=10
    )

    print(f"\nFound {len(results)} compounds tested against TP53:\n")
    for r in results[:5]:
        name = r.get("molecule_name") or r["molecule_chembl_id"]
        act_type = r.get("standard_type", "N/A")
        value = r.get("standard_value", "N/A")
        units = r.get("standard_units", "")
        print(f"  {name:<25} {act_type}={value} {units}")

    return results


if __name__ == "__main__":
    example_target_search()
    example_molecule_search()
    example_smiles_search()
    example_indication_search()
    example_uniprot_search()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
