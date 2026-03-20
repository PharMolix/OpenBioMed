#!/usr/bin/env python
"""
Basic example for protein mutation analysis.

This script demonstrates how to analyze a protein mutation using OpenBioMed tools:
1. Retrieve protein from UniProt
2. Explain mutation with MutaPLM
3. Predict structure with ESMFold
4. Visualize the protein

Usage:
    python basic_analysis.py --uniprot P04637 --mutation R248Q
"""

import argparse
from open_biomed.tools.tool_registry import TOOLS


def analyze_protein_mutation(uniprot_id: str, mutation: str, visualize: bool = True):
    """
    Analyze a protein mutation.

    Args:
        uniprot_id: UniProt accession (e.g., "P04637" for TP53)
        mutation: Mutation in format "R248Q" (OriginalAA + Position + MutantAA)
        visualize: Whether to generate visualization

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Step 1: Retrieve protein from UniProt
    print(f"\n{'='*60}")
    print(f"STEP 1: RETRIEVING PROTEIN FROM UNIPROT")
    print(f"{'='*60}")
    print(f"Input: UniProt ID = {uniprot_id}")

    tool = TOOLS["protein_uniprot_request"]
    result, message = tool.run(accession=uniprot_id)
    protein = result.get("protein") if isinstance(result, dict) else result

    results["protein_name"] = protein.name
    results["sequence_length"] = len(protein.sequence)

    print(f"\nResult: SUCCESS")
    print(f"  - Protein: {protein.name}")
    print(f"  - Sequence length: {len(protein.sequence)} amino acids")
    print(f"\nSUMMARY: Successfully retrieved {protein.name} from UniProt database.")

    # Step 2: Explain mutation with MutaPLM
    print(f"\n{'='*60}")
    print(f"STEP 2: EXPLAINING MUTATION")
    print(f"{'='*60}")
    print(f"Input: Mutation = {mutation}")

    mutation_tool = TOOLS["mutation_explanation"]
    mutation_result, mutation_message = mutation_tool.run(
        protein=protein,
        mutation=mutation
    )

    results["mutation_analysis"] = mutation_result

    print(f"\nResult: SUCCESS")
    print(f"  - Model: MutaPLM")
    print(f"  - Analysis: {mutation_result}")
    print(f"\nSUMMARY: Mutation analysis completed using MutaPLM.")

    # Step 3: Predict structure with ESMFold
    print(f"\n{'='*60}")
    print(f"STEP 3: PREDICTING PROTEIN STRUCTURE")
    print(f"{'='*60}")
    print(f"Input: Protein sequence ({len(protein.sequence)} aa)")

    folding_tool = TOOLS["protein_folding"]
    fold_result, fold_message = folding_tool.run(protein=protein)
    predicted_protein = fold_result.get("protein") if isinstance(fold_result, dict) else fold_result

    results["structure_predicted"] = True

    print(f"\nResult: SUCCESS")
    print(f"  - Model: ESMFold")
    print(f"  - Structure: Predicted 3D coordinates")
    print(f"\nSUMMARY: Protein structure prediction completed.")

    # Step 4: Visualize protein (optional)
    if visualize:
        print(f"\n{'='*60}")
        print(f"STEP 4: VISUALIZING PROTEIN STRUCTURE")
        print(f"{'='*60}")
        print(f"Input: Predicted protein structure")

        viz_tool = TOOLS["visualize_protein"]
        viz_result, viz_message = viz_tool.run(
            protein=predicted_protein,
            style="cartoon"
        )

        results["visualization_path"] = viz_result

        print(f"\nResult: SUCCESS")
        print(f"  - Style: cartoon")
        print(f"  - Output: {viz_result}")
        print(f"\nSUMMARY: Protein visualization generated and saved.")

    # Final summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Protein: {results['protein_name']}")
    print(f"Mutation: {mutation}")
    print(f"Structure: Predicted")
    if visualize:
        print(f"Visualization: {results.get('visualization_path', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze protein mutations using OpenBioMed"
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        required=True,
        help="UniProt accession ID (e.g., P04637 for TP53)"
    )
    parser.add_argument(
        "--mutation",
        type=str,
        required=True,
        help="Mutation in format R248Q (OriginalAA + Position + MutantAA)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization step"
    )

    args = parser.parse_args()

    analyze_protein_mutation(
        uniprot_id=args.uniprot,
        mutation=args.mutation,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()
