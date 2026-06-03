"""
Tools for reading molecule and protein file contents.
These tools are designed to help external agents access file content when they
cannot access the server's filesystem directly.

Usage:
- read_molecule_file: Read molecule from .pkl or .sdf file, return SMILES and SDF content
- read_protein_file: Read protein from .pkl or .pdb file, return FASTA sequence and PDB content
"""

from typing import Union, Tuple, List, Optional
import os
import pickle

from open_biomed.tools.base_tool import Tool
from open_biomed.data import Molecule, Protein


class ReadMoleculeFile(Tool):
    """
    Read molecule content from a file path.
    Supports .pkl (binary) and .sdf file formats.
    Returns SMILES string and optional SDF file content.
    """

    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Read molecule content from a file path
Inputs: {"molecule_file": str (path to .pkl or .sdf file), "include_sdf": bool (optional, whether to include SDF content)}
Outputs: {"smiles": str, "sdf_content": str (optional)}
"""

    def run(self, molecule_file: str, include_sdf: bool = True) -> Tuple[dict, str]:
        """
        Read molecule from file and return content.

        Args:
            molecule_file: Path to molecule file (.pkl or .sdf)
            include_sdf: Whether to include SDF content in output

        Returns:
            Tuple of (content_dict, message)
        """
        if not os.path.exists(molecule_file):
            raise FileNotFoundError(f"Molecule file not found: {molecule_file}")

        # Load molecule based on file extension
        if molecule_file.endswith(".pkl"):
            molecule = Molecule.from_binary_file(molecule_file)
        elif molecule_file.endswith(".sdf"):
            molecule = Molecule.from_sdf_file(molecule_file)
        else:
            raise ValueError(f"Unsupported file format: {molecule_file}")

        # Build content dictionary
        content = {
            "smiles": molecule.smiles,
            "name": molecule.name if hasattr(molecule, 'name') and molecule.name else "unknown"
        }

        # Include SDF content if requested
        if include_sdf:
            molecule._add_rdmol()
            molecule._add_conformer()
            # Generate SDF content
            from rdkit import Chem
            sdf_content = Chem.MolToMolBlock(molecule.rdmol)
            content["sdf_content"] = sdf_content

        message = f"Molecule content read from {molecule_file}: SMILES={molecule.smiles}"

        return content, message


class ReadProteinFile(Tool):
    """
    Read protein content from a file path.
    Supports .pkl (binary) and .pdb file formats.
    Returns FASTA sequence and optional PDB file content.
    """

    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Read protein content from a file path
Inputs: {"protein_file": str (path to .pkl or .pdb file), "include_pdb": bool (optional, whether to include PDB content)}
Outputs: {"sequence": str, "pdb_content": str (optional)}
"""

    def run(self, protein_file: str, include_pdb: bool = True) -> Tuple[dict, str]:
        """
        Read protein from file and return content.

        Args:
            protein_file: Path to protein file (.pkl or .pdb)
            include_pdb: Whether to include PDB content in output

        Returns:
            Tuple of (content_dict, message)
        """
        if not os.path.exists(protein_file):
            raise FileNotFoundError(f"Protein file not found: {protein_file}")

        # Load protein based on file extension
        if protein_file.endswith(".pkl"):
            protein = Protein.from_binary_file(protein_file)
        elif protein_file.endswith(".pdb"):
            protein = Protein.from_pdb_file(protein_file)
        else:
            raise ValueError(f"Unsupported file format: {protein_file}")

        # Build content dictionary
        content = {
            "sequence": protein.sequence if hasattr(protein, 'sequence') and protein.sequence else "",
            "name": protein.name if hasattr(protein, 'name') and protein.name else "unknown"
        }

        # Include PDB content if requested
        if include_pdb:
            # Check if protein has 3D structure (residues)
            has_structure = hasattr(protein, 'residues') and protein.residues is not None and len(protein.residues) > 0
            if has_structure:
                if protein_file.endswith(".pdb"):
                    # Read directly from file
                    with open(protein_file, "r") as f:
                        pdb_content = f.read()
                else:
                    # Save to temp file and read
                    temp_pdb = protein.save_pdb()
                    with open(temp_pdb, "r") as f:
                        pdb_content = f.read()
                content["pdb_content"] = pdb_content
            else:
                # Protein has no 3D structure, return empty PDB content with a note
                content["pdb_content"] = ""
                content["structure_note"] = "Protein has no 3D structure (only sequence available)"

        message = f"Protein content read from {protein_file}: sequence length={len(protein.sequence) if protein.sequence else 0}"

        return content, message


if __name__ == "__main__":
    # Test ReadMoleculeFile
    print("Testing ReadMoleculeFile...")
    # molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    # mol_file = molecule.save_binary()
    # tool = ReadMoleculeFile()
    # result = tool.run(mol_file)
    # print(result)

    # Test ReadProteinFile
    print("Testing ReadProteinFile...")
    # protein = Protein.from_fasta("MKFLILLFNILCLFPVLAADNH")
    # prot_file = protein.save_binary()
    # tool = ReadProteinFile()
    # result = tool.run(prot_file)
    # print(result)