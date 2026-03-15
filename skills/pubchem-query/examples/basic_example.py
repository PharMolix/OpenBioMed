#!/usr/bin/env python
"""
Basic examples for PubChem Query skill.

Usage:
    python basic_example.py
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def example_name_to_structure():
    """Example 1: Convert drug name to structure."""
    print("=" * 60)
    print("Example 1: Name to Structure")
    print("=" * 60)

    from open_biomed.tools.tool_registry import TOOLS

    tool = TOOLS["molecule_name_request"]

    # Query by name
    molecules, _ = tool.run("aspirin")
    mol = molecules[0]

    print(f"Input: aspirin")
    print(f"SMILES: {mol.smiles}")
    print(f"SDF saved: ./tmp/pubchem_aspirin.sdf")
    print()


def example_similarity_search():
    """Example 2: Find similar compounds."""
    print("=" * 60)
    print("Example 2: Similarity Search")
    print("=" * 60)

    from open_biomed.data import Molecule
    from open_biomed.tools.tool_registry import TOOLS

    tool = TOOLS["molecule_structure_request"]

    # Query with aspirin SMILES
    query_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    query = Molecule.from_smiles(query_smiles)

    molecules, _ = tool.run(molecule=query, threshold=0.85, max_records=5)

    print(f"Query SMILES: {query_smiles}")
    print(f"Threshold: 85%")
    print(f"Found {len(molecules)} similar compounds:")
    for i, mol in enumerate(molecules):
        print(f"  {i+1}. {mol.smiles}")
    print()


def example_bioactivity_compound():
    """Example 3a: Get assays where a compound was active."""
    print("=" * 60)
    print("Example 3a: Bioactivity Query - Compound")
    print("=" * 60)

    from open_biomed.tools.tool_registry import TOOLS

    tool = TOOLS["pubchem_bioactivity"]

    # Get assays where aspirin (CID 2244) was active
    results, _ = tool.run(query_type="compound", cid=2244, aids_type="active")

    print(f"Query: Assays where aspirin (CID 2244) was active")
    print(f"Found {len(results)} assays")
    print("First 5 assays:")
    for r in results[:5]:
        print(f"  AID: {r['AID']}")
    print()


def example_bioactivity_assay():
    """Example 3b: Get compounds active in an assay."""
    print("=" * 60)
    print("Example 3b: Bioactivity Query - Assay")
    print("=" * 60)

    from open_biomed.tools.tool_registry import TOOLS

    tool = TOOLS["pubchem_bioactivity"]

    # Get compounds active in assay 1195
    results, _ = tool.run(query_type="assay", aid=1195, cids_type="active")

    print(f"Query: Compounds active in assay 1195")
    print(f"Found {len(results)} compounds")
    print("First 5 compounds:")
    for r in results[:5]:
        print(f"  CID: {r['CID']}")
    print()


def example_bioactivity_target():
    """Example 3c: Get assays targeting a gene."""
    print("=" * 60)
    print("Example 3c: Bioactivity Query - Target")
    print("=" * 60)

    from open_biomed.tools.tool_registry import TOOLS

    tool = TOOLS["pubchem_bioactivity"]

    # Get assays targeting PTGS2 (COX-2 gene)
    results, _ = tool.run(query_type="target", gene_symbol="PTGS2")

    print(f"Query: Assays targeting PTGS2 (COX-2)")
    print(f"Found {len(results)} assays")
    print("First 5 assays:")
    for r in results[:5]:
        print(f"  AID: {r['AID']}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PubChem Query Skill - Basic Examples")
    print("=" * 60 + "\n")

    try:
        example_name_to_structure()
        example_similarity_search()
        example_bioactivity_compound()
        example_bioactivity_assay()
        example_bioactivity_target()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
