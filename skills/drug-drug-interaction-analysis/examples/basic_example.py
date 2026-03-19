#!/usr/bin/env python
"""
Drug-Drug Interaction Analysis Example

This script demonstrates how to analyze potential drug-drug interactions
using the KEGG DDI database for up to 5 drugs.
"""

import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time


KEGG_API_BASE = "https://rest.kegg.jp"


@dataclass
class DrugInfo:
    """Drug information container."""
    kegg_id: str
    name: str
    formula: str = ""
    targets: List[str] = None
    enzymes: List[str] = None
    atc_codes: List[str] = None

    def __post_init__(self):
        if self.targets is None:
            self.targets = []
        if self.enzymes is None:
            self.enzymes = []
        if self.atc_codes is None:
            self.atc_codes = []


@dataclass
class Interaction:
    """Drug-drug interaction result."""
    drug_a_id: str
    drug_b_id: str
    severity_code: str
    mechanism: str

    @property
    def severity(self) -> str:
        """Human-readable severity level."""
        severity_map = {
            "CI": "Contraindicated",
            "P": "Precaution",
            "C": "Caution"
        }
        return severity_map.get(self.severity_code.split(",")[0], "Unknown")

    def format_mechanism(self) -> str:
        """Format mechanism for display."""
        if not self.mechanism:
            return "Not specified"
        return self.mechanism


class DDIAnalyzer:
    """Drug-Drug Interaction Analyzer using KEGG DDI database."""

    def __init__(self, rate_limit_delay: float = 0.2):
        """
        Initialize the analyzer.

        Args:
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.rate_limit_delay = rate_limit_delay
        self._drug_cache: Dict[str, DrugInfo] = {}

    def _kegg_request(self, operation: str, *args) -> str:
        """Make a KEGG API request with rate limiting."""
        url = f"{KEGG_API_BASE}/{operation}"
        if args:
            url += "/" + "/".join(str(a) for a in args)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        time.sleep(self.rate_limit_delay)
        return response.text

    def find_drug_id(self, drug_name: str) -> Optional[str]:
        """
        Find KEGG drug ID from drug name.

        Args:
            drug_name: Common drug name (e.g., "aspirin", "ibuprofen")

        Returns:
            KEGG drug ID (e.g., "D00109") or None if not found
        """
        result = self._kegg_request("find", "drug", drug_name)
        if result.strip():
            first_line = result.strip().split('\n')[0]
            kegg_id = first_line.split('\t')[0]
            # Extract just the ID part (e.g., "D00109" from "dr:D00109")
            return kegg_id.replace("dr:", "")
        return None

    def get_drug_info(self, drug_id: str) -> DrugInfo:
        """
        Get detailed drug information from KEGG.

        Args:
            drug_id: KEGG drug ID (e.g., "D00109")

        Returns:
            DrugInfo object with drug details
        """
        if drug_id in self._drug_cache:
            return self._drug_cache[drug_id]

        result = self._kegg_request("get", f"dr:{drug_id}")

        name = ""
        formula = ""
        targets = []
        enzymes = []
        atc_codes = []

        for line in result.split('\n'):
            if line.startswith("NAME"):
                name = line.split(maxsplit=1)[1].strip().rstrip(";")
            elif line.startswith("FORMULA"):
                formula = line.split(maxsplit=1)[1].strip()
            elif line.startswith("TARGET"):
                targets.append(line.split(maxsplit=1)[1].strip())
            elif line.startswith("METABOLISM"):
                enzymes.append(line.split(maxsplit=1)[1].strip())
            elif line.startswith("REMARK"):
                if "ATC code:" in line:
                    atc_str = line.split("ATC code:")[1].strip()
                    atc_codes.extend(atc_str.split())

        drug_info = DrugInfo(
            kegg_id=drug_id,
            name=name,
            formula=formula,
            targets=targets,
            enzymes=enzymes,
            atc_codes=atc_codes
        )
        self._drug_cache[drug_id] = drug_info
        return drug_info

    def get_interactions(self, drug_ids: List[str]) -> List[Interaction]:
        """
        Query KEGG DDI for drug-drug interactions.

        Args:
            drug_ids: List of KEGG drug IDs

        Returns:
            List of Interaction objects
        """
        ids = "+".join(drug_ids)
        result = self._kegg_request("ddi", ids)

        interactions = []
        for line in result.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                interactions.append(Interaction(
                    drug_a_id=parts[0].replace("dr:", ""),
                    drug_b_id=parts[1].replace("dr:", ""),
                    severity_code=parts[2],
                    mechanism=parts[3] if len(parts) > 3 else ""
                ))
        return interactions

    def analyze(self, drugs: List[str]) -> Dict:
        """
        Perform complete DDI analysis for multiple drugs.

        Args:
            drugs: List of drug names (up to 5)

        Returns:
            Dictionary with analysis results
        """
        if len(drugs) < 2:
            return {"error": "At least 2 drugs required for DDI analysis"}
        if len(drugs) > 5:
            return {"error": "Maximum 5 drugs allowed per analysis"}

        # Step 1: Resolve drug names to KEGG IDs
        drug_ids = {}
        unresolved = []

        for drug in drugs:
            kegg_id = self.find_drug_id(drug)
            if kegg_id:
                drug_ids[drug] = kegg_id
            else:
                unresolved.append(drug)

        if not drug_ids:
            return {"error": "No drugs could be resolved", "unresolved": unresolved}

        # Step 2: Get drug information
        drug_info = {}
        for drug, kegg_id in drug_ids.items():
            drug_info[drug] = self.get_drug_info(kegg_id)

        # Step 3: Query DDI
        interactions = self.get_interactions(list(drug_ids.values()))

        # Step 4: Build results
        results = []
        for interaction in interactions:
            drug_a_name = next(
                (d for d, info in drug_info.items() if info.kegg_id == interaction.drug_a_id),
                interaction.drug_a_id
            )
            drug_b_name = next(
                (d for d, info in drug_info.items() if info.kegg_id == interaction.drug_b_id),
                interaction.drug_b_id
            )

            results.append({
                "drug_a": f"{drug_a_name} ({interaction.drug_a_id})",
                "drug_b": f"{drug_b_name} ({interaction.drug_b_id})",
                "severity": interaction.severity,
                "severity_code": interaction.severity_code,
                "mechanism": interaction.format_mechanism(),
            })

        # Calculate statistics
        severity_counts = {"Contraindicated": 0, "Precaution": 0, "Caution": 0}
        for r in results:
            severity_counts[r["severity"]] = severity_counts.get(r["severity"], 0) + 1

        return {
            "drugs_analyzed": list(drug_ids.keys()),
            "drug_ids": drug_ids,
            "total_pairs": len(drugs) * (len(drugs) - 1) // 2,
            "interactions_found": len(results),
            "severity_summary": severity_counts,
            "interactions": results,
            "drug_details": {
                drug: {
                    "name": info.name,
                    "formula": info.formula,
                    "targets": info.targets,
                    "enzymes": info.enzymes,
                    "atc_codes": info.atc_codes
                }
                for drug, info in drug_info.items()
            },
            "unresolved": unresolved
        }


def main():
    """Run DDI analysis example."""
    print("=" * 60)
    print("Drug-Drug Interaction Analysis")
    print("=" * 60)

    # Example 1: Aspirin + Ibuprofen
    print("\n--- Example 1: Aspirin + Ibuprofen ---")
    analyzer = DDIAnalyzer()
    result = analyzer.analyze(["aspirin", "ibuprofen"])

    print(f"Drugs analyzed: {result.get('drugs_analyzed', [])}")
    print(f"Interactions found: {result.get('interactions_found', 0)}")

    for interaction in result.get("interactions", []):
        print(f"\n  {interaction['drug_a']} + {interaction['drug_b']}")
        print(f"    Severity: {interaction['severity']}")
        print(f"    Mechanism: {interaction['mechanism']}")

    # Example 2: More complex interaction
    print("\n--- Example 2: Warfarin + Fluconazole ---")
    result2 = analyzer.analyze(["warfarin", "fluconazole"])

    print(f"Drugs analyzed: {result2.get('drugs_analyzed', [])}")
    print(f"Interactions found: {result2.get('interactions_found', 0)}")

    for interaction in result2.get("interactions", []):
        print(f"\n  {interaction['drug_a']} + {interaction['drug_b']}")
        print(f"    Severity: {interaction['severity']}")
        print(f"    Mechanism: {interaction['mechanism']}")

    # Example 3: Multiple drugs
    print("\n--- Example 3: Multiple Drug Analysis ---")
    result3 = analyzer.analyze(["aspirin", "warfarin", "omeprazole"])

    print(f"Drugs analyzed: {result3.get('drugs_analyzed', [])}")
    print(f"Total pairs checked: {result3.get('total_pairs', 0)}")
    print(f"Interactions found: {result3.get('interactions_found', 0)}")
    print(f"Severity summary: {result3.get('severity_summary', {})}")

    for interaction in result3.get("interactions", []):
        print(f"\n  {interaction['drug_a']} + {interaction['drug_b']}")
        print(f"    Severity: {interaction['severity']}")
        print(f"    Mechanism: {interaction['mechanism']}")


if __name__ == "__main__":
    main()
