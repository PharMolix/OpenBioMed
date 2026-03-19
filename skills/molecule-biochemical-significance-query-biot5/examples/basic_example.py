#!/usr/bin/env python
"""
Basic example for molecule biochemical significance query using BioT5.

This script demonstrates how to query a molecule's biochemical significance
and roles in biology and chemistry using the molecule_question_answering tool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from open_biomed.data import Molecule, Text
from open_biomed.tools.tool_registry import TOOLS


def query_biochemical_significance_from_smiles(smiles: str) -> str:
    """
    Query a molecule's biochemical significance from its SMILES string.

    Args:
        smiles: SMILES string representation of the molecule

    Returns:
        Natural language description of the molecule's biochemical significance
    """
    # Step 1: Create molecule from SMILES
    molecule = Molecule.from_smiles(smiles)
    print(f"SMILES: {molecule.smiles}")
    print(f"Number of atoms: {molecule.get_num_atoms()}")

    # Step 2: Ask about biochemical significance
    qa_tool = TOOLS["molecule_question_answering"]
    question = Text.from_str(
        "I am interested in understanding the molecule biochemical significance; "
        "can you describe its roles in biology and chemistry?"
    )

    outputs, message = qa_tool.run(molecule=molecule, text=question)

    return outputs[0]


def query_biochemical_significance_from_name(name: str) -> str:
    """
    Query a molecule's biochemical significance from its common name.

    Args:
        name: Common name of the molecule (e.g., "aspirin", "caffeine")

    Returns:
        Natural language description of the molecule's biochemical significance
    """
    # Step 1: Get molecule from name via PubChem
    name_tool = TOOLS["molecule_name_request"]
    molecules, message = name_tool.run(name)

    if not molecules:
        raise ValueError(f"Could not find molecule: {name}")

    molecule = molecules[0]
    print(f"Name: {name}")
    print(f"SMILES: {molecule.smiles}")

    # Step 2: Ask about biochemical significance
    qa_tool = TOOLS["molecule_question_answering"]
    question = Text.from_str(
        "I am interested in understanding the molecule biochemical significance; "
        "can you describe its roles in biology and chemistry?"
    )

    outputs, message = qa_tool.run(molecule=molecule, text=question)

    return outputs[0]


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Query from SMILES")
    print("=" * 60)

    smiles = "CCCCCCCc1ccco1"  # Heptylfuran
    result = query_biochemical_significance_from_smiles(smiles)
    print(f"\nBiochemical Significance: {result}")

    print("\n" + "=" * 60)
    print("Example 2: Query from molecule name")
    print("=" * 60)

    name = "aspirin"
    result = query_biochemical_significance_from_name(name)
    print(f"\nBiochemical Significance: {result}")
