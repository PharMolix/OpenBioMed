#!/usr/bin/env python3
"""
Basic example: Query STRING database for protein-protein interactions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from open_biomed.tools.tool_registry import TOOLS


def query_string_interactions(uniprot_id: str, required_score: int = 700, limit: int = 20):
    """
    Query STRING for protein interaction partners.

    Args:
        uniprot_id: UniProt accession (e.g., P04637 for TP53)
        required_score: Minimum confidence (150/400/700/900)
        limit: Maximum number of interactors

    Returns:
        List of interaction records
    """
    tool = TOOLS["ppi_string_request"]
    results, _ = tool.run(
        uniprot_id=uniprot_id,
        species=9606,
        required_score=required_score,
        limit=limit
    )
    return results


def format_interaction_report(results: list, query_protein: str) -> str:
    """Format interaction results as a readable report."""
    lines = [
        "=" * 70,
        f"STRING INTERACTION REPORT: {query_protein}",
        "=" * 70,
        f"Total interactions found: {len(results)}",
        "",
        f"{'Partner':<12} {'Combined':<10} {'Exp':<8} {'Text':<8} {'DB':<8}",
        "-" * 70
    ]

    for interaction in results:
        partner = interaction['partner_gene']
        combined = interaction['combined_score']
        exp = interaction['scores']['experimental'] or 0
        text = interaction['scores']['text_mining'] or 0
        db = interaction['scores']['database'] or 0

        lines.append(f"{partner:<12} {combined:<10.3f} {exp:<8.3f} {text:<8.3f} {db:<8.3f}")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    # Example 1: Query TP53 interactions
    print("\n" + "=" * 70)
    print("Example 1: TP53 (P04637) - Tumor Suppressor p53")
    print("=" * 70)

    results = query_string_interactions("P04637", required_score=700, limit=15)
    print(format_interaction_report(results, "TP53"))

    # Example 2: Query EGFR interactions
    print("\n" + "=" * 70)
    print("Example 2: EGFR (P00533) - Epidermal Growth Factor Receptor")
    print("=" * 70)

    results = query_string_interactions("P00533", required_score=700, limit=15)
    print(format_interaction_report(results, "EGFR"))

    # Example 3: Higher confidence threshold
    print("\n" + "=" * 70)
    print("Example 3: CDK2 (P24941) - Highest confidence only (900)")
    print("=" * 70)

    results = query_string_interactions("P24941", required_score=900, limit=10)
    print(format_interaction_report(results, "CDK2"))
