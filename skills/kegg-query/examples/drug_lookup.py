#!/usr/bin/env python3
"""
Drug Information Lookup Example

Demonstrates how to retrieve comprehensive drug information from KEGG DRUG database.
"""

import requests
from typing import Dict, List, Optional, Tuple


# KEGG API base URL
KEGG_API_BASE = "https://rest.kegg.jp"


def kegg_find(database: str, query: str, option: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Search KEGG database for entries matching query.

    Args:
        database: KEGG database name (drug, compound, disease, pathway, genes, etc.)
        query: Search keyword or identifier
        option: Optional search mode (formula, exact_mass, mol_weight for compound/drug)

    Returns:
        List of (entry_id, description) tuples
    """
    url = f"{KEGG_API_BASE}/find/{database}/{query}"
    if option:
        url += f"/{option}"

    response = requests.get(url)
    response.raise_for_status()

    results = []
    for line in response.text.strip().split("\n"):
        if line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                results.append((parts[0], parts[1]))

    return results


def kegg_get(entry_id: str, option: Optional[str] = None) -> str:
    """
    Retrieve KEGG entry by ID.

    Args:
        entry_id: KEGG entry identifier (e.g., D00109, hsa00010, H00409)
        option: Optional format (aaseq, ntseq, mol, image, kgml)

    Returns:
        Raw KEGG entry text
    """
    # Ensure proper ID format
    if not any(entry_id.startswith(p) for p in ["dr:", "cpd:", "ds:", "hsa:", "map:", "ko:"]):
        # Auto-detect database from ID format
        if entry_id.startswith("D"):
            entry_id = f"dr:{entry_id}"
        elif entry_id.startswith("C"):
            entry_id = f"cpd:{entry_id}"
        elif entry_id.startswith("H"):
            entry_id = f"ds:{entry_id}"

    url = f"{KEGG_API_BASE}/get/{entry_id}"
    if option:
        url += f"/{option}"

    response = requests.get(url)
    response.raise_for_status()

    return response.text


def parse_drug_entry(entry: str) -> Dict:
    """
    Parse KEGG DRUG entry into structured dictionary.

    Args:
        entry: Raw KEGG entry text

    Returns:
        Parsed drug information dictionary
    """
    drug = {
        "id": "",
        "names": [],
        "formula": "",
        "exact_mass": "",
        "mol_weight": "",
        "efficacy": [],
        "diseases": [],
        "targets": [],
        "pathways": [],
        "interactions": [],
        "class_hierarchy": [],
        "external_ids": {},
        "atom_structure": [],
    }

    current_field = None
    lines = entry.strip().split("\n")

    for i, line in enumerate(lines):
        if not line:
            continue

        # Check for field name at start of line
        if len(line) > 12 and line[12] != " ":
            field_name = line[:12].strip()
            field_value = line[12:].strip()
            current_field = field_name

            if field_name == "ENTRY":
                drug["id"] = field_value.split()[0]
            elif field_name == "NAME":
                drug["names"] = [n.strip().rstrip(";") for n in field_value.split(";")]
            elif field_name == "FORMULA":
                drug["formula"] = field_value
            elif field_name == "EXACT_MASS":
                drug["exact_mass"] = field_value
            elif field_name == "MOL_WEIGHT":
                drug["mol_weight"] = field_value
            elif field_name == "EFFICACY":
                drug["efficacy"] = [e.strip() for e in field_value.split(",")]
            elif field_name == "TARGET":
                drug["targets"].append(_parse_target(field_value))
            elif field_name == "PATHWAY":
                parts = field_value.split(None, 1)
                if len(parts) == 2:
                    drug["pathways"].append({"id": parts[0].strip("()"), "name": parts[1]})
            elif field_name == "DBLINKS":
                if ":" in field_value:
                    db, val = field_value.split(":", 1)
                    drug["external_ids"][db.strip()] = val.strip()
            elif field_name == "REMARK":
                drug["remarks"] = field_value
        elif current_field in ["NAME", "TARGET", "PATHWAY", "EFFICACY"]:
            # Continuation of previous field
            value = line.strip()
            if current_field == "NAME":
                drug["names"].extend([n.strip().rstrip(";") for n in value.split(";")])
            elif current_field == "TARGET":
                drug["targets"].append(_parse_target(value))
            elif current_field == "PATHWAY":
                parts = value.split(None, 1)
                if len(parts) == 2:
                    drug["pathways"].append({"id": parts[0].strip("()"), "name": parts[1]})
            elif current_field == "EFFICACY":
                drug["efficacy"].extend([e.strip() for e in value.split(",")])
        elif current_field == "DISEASE":
            # Parse disease entries under EFFICACY
            value = line.strip()
            if value.startswith("[DS:"):
                disease_id = value.split("]")[0].replace("[DS:", "")
                disease_name = value.split("]", 1)[1].strip() if "]" in value else ""
                drug["diseases"].append({"id": disease_id, "name": disease_name})

    return drug


def _parse_target(target_str: str) -> Dict:
    """Parse target string into structured format."""
    target = {"raw": target_str}

    # Extract gene symbol
    if "(" in target_str:
        target["gene"] = target_str.split("(")[0].strip()

    # Extract UniProt ID from [HSA:xxxxx]
    if "[HSA:" in target_str:
        import re
        match = re.search(r"\[HSA:(\d+)\]", target_str)
        if match:
            target["entrez_id"] = match.group(1)

    # Extract KO ID from [KO:Kxxxxx]
    if "[KO:" in target_str:
        import re
        match = re.search(r"\[KO:(K\d+)\]", target_str)
        if match:
            target["ko"] = match.group(1)

    return target


def format_drug_report(drug: Dict) -> str:
    """Format drug information as human-readable report."""
    lines = [
        "=" * 70,
        "KEGG DRUG REPORT",
        "=" * 70,
        f"Drug ID:      {drug['id']}",
        f"Names:        {', '.join(drug['names'][:3])}",
        f"Formula:      {drug['formula']}",
        f"Mol. Weight:  {drug['mol_weight']}",
        "",
    ]

    if drug["efficacy"]:
        lines.extend([
            "EFFICACY",
            "-" * 40,
            ", ".join(drug["efficacy"]),
            "",
        ])

    if drug["targets"]:
        lines.extend([
            "TARGETS",
            "-" * 40,
        ])
        for target in drug["targets"][:5]:
            gene = target.get("gene", "Unknown")
            ko = target.get("ko", "")
            lines.append(f"  • {gene}" + (f" [KO:{ko}]" if ko else ""))
        lines.append("")

    if drug["pathways"]:
        lines.extend([
            "PATHWAYS",
            "-" * 40,
        ])
        for pw in drug["pathways"][:5]:
            lines.append(f"  • {pw['id']}: {pw['name']}")
        lines.append("")

    if drug["diseases"]:
        lines.extend([
            "DISEASES",
            "-" * 40,
        ])
        for disease in drug["diseases"][:5]:
            lines.append(f"  • {disease['name']} [{disease['id']}]")
        lines.append("")

    if drug["external_ids"]:
        lines.extend([
            "EXTERNAL DATABASES",
            "-" * 40,
        ])
        for db, val in drug["external_ids"].items():
            lines.append(f"  • {db}: {val}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Example: Look up aspirin in KEGG DRUG database."""
    print("Searching for 'aspirin' in KEGG DRUG database...\n")

    # Step 1: Find drug ID
    results = kegg_find("drug", "aspirin")
    print(f"Found {len(results)} matches:")
    for entry_id, description in results[:5]:
        print(f"  {entry_id}: {description[:60]}...")

    # Step 2: Get full entry for first match (D00109)
    print("\n" + "=" * 70)
    print("Fetching full entry for D00109 (Aspirin)...\n")

    entry = kegg_get("D00109")
    drug = parse_drug_entry(entry)

    # Print formatted report
    print(format_drug_report(drug))

    # Print as JSON for programmatic use
    print("\nStructured data (JSON):")
    import json
    print(json.dumps({
        "id": drug["id"],
        "names": drug["names"],
        "formula": drug["formula"],
        "efficacy": drug["efficacy"],
        "targets": drug["targets"],
        "pathways": drug["pathways"],
    }, indent=2))


if __name__ == "__main__":
    main()
