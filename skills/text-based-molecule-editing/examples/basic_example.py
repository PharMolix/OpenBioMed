"""
Basic example for text-based molecule editing.

This script demonstrates how to modify a molecule based on natural language
descriptions using the MolT5/BioT5 model.
"""

import sys
sys.path.insert(0, '.')

from open_biomed.data import Molecule, Text
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.core.pipeline import InferencePipeline


def edit_molecule_by_name(name: str, edit_prompt: str) -> dict:
    """
    Edit a molecule by its common name with a text description.

    Args:
        name: Common molecule name (e.g., "aspirin", "ibuprofen")
        edit_prompt: Natural language description of desired changes

    Returns:
        Dictionary with original and edited molecule info
    """
    # Step 1: Get molecule from name
    tool = TOOLS["molecule_name_request"]
    result, _ = tool.run(accession=name)
    molecule = result[0]

    # Step 2: Calculate baseline properties
    qed_tool = TOOLS["molecule_qed"]
    logp_tool = TOOLS["molecule_logp"]
    sa_tool = TOOLS["molecule_sa"]

    qed_orig, _ = qed_tool.run(molecule=molecule)
    logp_orig, _ = logp_tool.run(molecule=molecule)
    sa_orig, _ = sa_tool.run(molecule=molecule)

    # Step 3: Run text-based editing
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device="cuda:0"
    )

    outputs = pipeline.run(
        molecule=molecule,
        text=Text.from_str(edit_prompt),
    )
    edited_molecule = outputs[0][0]

    # Step 4: Calculate new properties
    qed_new, _ = qed_tool.run(molecule=edited_molecule)
    logp_new, _ = logp_tool.run(molecule=edited_molecule)
    sa_new, _ = sa_tool.run(molecule=edited_molecule)

    return {
        "original": {
            "smiles": molecule.smiles,
            "qed": qed_orig[0],
            "logp": logp_orig[0],
            "sa": sa_orig[0],
        },
        "edited": {
            "smiles": edited_molecule.smiles,
            "qed": qed_new[0],
            "logp": logp_new[0],
            "sa": sa_new[0],
        },
        "prompt": edit_prompt,
    }


def edit_molecule_by_smiles(smiles: str, edit_prompt: str) -> dict:
    """
    Edit a molecule from SMILES string with a text description.

    Args:
        smiles: SMILES string of input molecule
        edit_prompt: Natural language description of desired changes

    Returns:
        Dictionary with original and edited molecule info
    """
    molecule = Molecule.from_smiles(smiles)

    # Same workflow as edit_molecule_by_name
    qed_tool = TOOLS["molecule_qed"]
    logp_tool = TOOLS["molecule_logp"]
    sa_tool = TOOLS["molecule_sa"]

    qed_orig, _ = qed_tool.run(molecule=molecule)
    logp_orig, _ = logp_tool.run(molecule=molecule)
    sa_orig, _ = sa_tool.run(molecule=molecule)

    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device="cuda:0"
    )

    outputs = pipeline.run(
        molecule=molecule,
        text=Text.from_str(edit_prompt),
    )
    edited_molecule = outputs[0][0]

    qed_new, _ = qed_tool.run(molecule=edited_molecule)
    logp_new, _ = logp_tool.run(molecule=edited_molecule)
    sa_new, _ = sa_tool.run(molecule=edited_molecule)

    return {
        "original": {
            "smiles": molecule.smiles,
            "qed": qed_orig[0],
            "logp": logp_orig[0],
            "sa": sa_orig[0],
        },
        "edited": {
            "smiles": edited_molecule.smiles,
            "qed": qed_new[0],
            "logp": logp_new[0],
            "sa": sa_new[0],
        },
        "prompt": edit_prompt,
    }


def print_comparison(result: dict):
    """Print a formatted comparison of original vs edited molecule."""
    print("=" * 60)
    print("MOLECULE EDITING RESULT")
    print("=" * 60)
    print(f"Prompt: {result['prompt']}")
    print()
    print(f"{'Property':<20} {'Original':<15} {'Edited':<15} {'Change':<15}")
    print("-" * 65)

    orig = result['original']
    edit = result['edited']

    print(f"{'SMILES':<20} {orig['smiles'][:15]:<15} {edit['smiles'][:15]:<15} {'-':<15}")
    print(f"{'QED':<20} {orig['qed']:<15.4f} {edit['qed']:<15.4f} {edit['qed']-orig['qed']:+.4f}")
    print(f"{'LogP':<20} {orig['logp']:<15.4f} {edit['logp']:<15.4f} {edit['logp']-orig['logp']:+.4f}")
    print(f"{'SA':<20} {orig['sa']:<15.4f} {edit['sa']:<15.4f} {edit['sa']-orig['sa']:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # Example 1: Edit by name
    result = edit_molecule_by_name(
        name="aspirin",
        edit_prompt="This molecule should be more soluble in water"
    )
    print_comparison(result)

    # Example 2: Edit by SMILES
    result = edit_molecule_by_smiles(
        smiles="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        edit_prompt="This molecule should have better drug-likeness"
    )
    print_comparison(result)
