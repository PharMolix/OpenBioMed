#!/usr/bin/env python
"""
Basic example for IUPAC name identification using MolT5/BioT5.

Usage:
    python basic_example.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
    python basic_example.py --name "aspirin"
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from open_biomed.data import Molecule, Text
from open_biomed.tools.tool_registry import TOOLS


def get_iupac_from_smiles(smiles: str, question: str = "What's the IUPAC name of this molecule?") -> str:
    """
    Get IUPAC name from a SMILES string.

    Args:
        smiles: SMILES string of the molecule
        question: Question to ask the model

    Returns:
        IUPAC name string
    """
    # Create molecule from SMILES
    molecule = Molecule.from_smiles(smiles)

    # Create question text
    text = Text.from_str(question)

    # Use molecule question answering tool
    qa_tool = TOOLS["molecule_question_answering"]
    result, message = qa_tool.run(molecule=molecule, text=text)

    return result


def get_iupac_from_name(name: str, question: str = "What's the IUPAC name of this molecule?") -> str:
    """
    Get IUPAC name from a common molecule name.

    Args:
        name: Common name of the molecule (e.g., "aspirin")
        question: Question to ask the model

    Returns:
        IUPAC name string
    """
    # Retrieve molecule from PubChem
    name_tool = TOOLS["molecule_name_request"]
    mol_result, mol_message = name_tool.run(accession=name)
    molecule = mol_result[0]  # Returns a list of molecules

    print(f"Retrieved molecule: {mol_message}")

    # Create question text
    text = Text.from_str(question)

    # Use molecule question answering tool
    qa_tool = TOOLS["molecule_question_answering"]
    result, message = qa_tool.run(molecule=molecule, text=text)

    return result


def main():
    parser = argparse.ArgumentParser(description="Identify IUPAC name of a molecule")
    parser.add_argument("--smiles", type=str, help="SMILES string of the molecule")
    parser.add_argument("--name", type=str, help="Common name of the molecule")
    parser.add_argument("--question", type=str, default="What's the IUPAC name of this molecule?",
                        help="Question to ask the model")
    args = parser.parse_args()

    if not args.smiles and not args.name:
        parser.error("Please provide either --smiles or --name")

    if args.smiles:
        print(f"SMILES: {args.smiles}")
        iupac = get_iupac_from_smiles(args.smiles, args.question)
    else:
        print(f"Name: {args.name}")
        iupac = get_iupac_from_name(args.name, args.question)

    print(f"\nIUPAC Name: {iupac}")


if __name__ == "__main__":
    main()
