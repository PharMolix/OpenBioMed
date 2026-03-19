#!/usr/bin/env python
"""
Basic example for protein function annotation using BioT5.

This script demonstrates how to predict protein function and properties
from an amino acid sequence.

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


def annotate_protein_function(sequence: str) -> str:
    """
    Annotate protein function from amino acid sequence.

    Args:
        sequence: Amino acid sequence in FASTA format (single letter codes)

    Returns:
        Functional annotation text describing the protein's properties
    """
    # Create protein object from sequence
    protein = Protein.from_fasta(sequence)

    # Create the standard question for functional annotation
    question = Text.from_str(
        "Inspect the protein sequence and offer a concise description of its properties."
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
    # This enzyme is involved in purine biosynthesis
    sequence = (
        "MRVGVIRFPGSNCDRDVHHVLELAGAEPEYVWWNQRNLDHLDAVVIPGGFSYGDYLRAGAIAAITPVMDAVRELVRE"
        "EKPVLGICNGAQILAEVGLVPGVFTVNEHPKFNCQWTELRVKTTRTPFTGLFKKDEVIRMPVAHAEGRYYHDNISEVW"
        "ENDQVVLQFHGENPNGSLDGITGVCDESGLVCAVMPHPERASEEILGSVDGFKFFRGILKFRG"
    )

    print("=" * 60)
    print("Protein Function Annotation")
    print("=" * 60)
    print(f"\nInput sequence length: {len(sequence)} amino acids")
    print(f"First 50 residues: {sequence[:50]}...")
    print()

    print("Running BioT5 model for function prediction...")
    print("-" * 60)

    annotation = annotate_protein_function(sequence)

    print("\nFunctional Annotation:")
    print("-" * 60)
    print(annotation)
    print("-" * 60)


if __name__ == "__main__":
    main()
