#!/usr/bin/env python
"""
Basic example for protein subcellular localization prediction using BioT5.

This script demonstrates how to predict where a protein localizes in the cell
from its amino acid sequence.

Usage:
    python examples/basic_example.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['dataset_name'] = 'BBBP'

from open_biomed.data import Protein, Text
from open_biomed.core.pipeline import InferencePipeline


def predict_subcellular_localization(sequence: str) -> str:
    """
    Predict subcellular localization from amino acid sequence.

    Args:
        sequence: Amino acid sequence in FASTA format (single letter codes)

    Returns:
        Subcellular localization prediction (e.g., "Cytoplasm", "Nucleus")
    """
    # Create protein object from sequence
    protein = Protein.from_fasta(sequence)

    # Create the standard question for localization prediction
    question = Text.from_str(
        "Please provide information about the subcellular localization of this protein."
    )

    # Load the BioT5 model for protein question answering
    pipeline = InferencePipeline(
        task="protein_question_answering",
        model="biot5",
        model_ckpt="./checkpoints/server/protein_question_answering_biot5.ckpt",
        device="cuda:0"
    )

    # Run inference
    outputs = pipeline.run(protein=protein, text=question)

    return outputs[0][0].str


def main():
    # Example: Phosphoribosylformylglycinamidine synthase subunit PurQ
    # This is a cytoplasmic enzyme involved in purine biosynthesis
    sequence = (
        "MRVGVIRFPGSNCDRDVHHVLELAGAEPEYVWWNQRNLDHLDAVVIPGGFSYGDYLRAGAIAAITPVMDAVRELVRE"
        "EKPVLGICNGAQILAEVGLVPGVFTVNEHPKFNCQWTELRVKTTRTPFTGLFKKDEVIRMPVAHAEGRYYHDNISEVW"
        "ENDQVVLQFHGENPNGSLDGITGVCDESGLVCAVMPHPERASEEILGSVDGFKFFRGILKFRG"
    )

    print("=" * 60)
    print("Protein Subcellular Localization Prediction")
    print("=" * 60)
    print(f"\nInput sequence length: {len(sequence)} amino acids")
    print(f"First 50 residues: {sequence[:50]}...")
    print()

    print("Running BioT5 model for localization prediction...")
    print("-" * 60)

    localization = predict_subcellular_localization(sequence)

    print("\nSubcellular Localization:")
    print("-" * 60)
    print(localization)
    print("-" * 60)


if __name__ == "__main__":
    main()
