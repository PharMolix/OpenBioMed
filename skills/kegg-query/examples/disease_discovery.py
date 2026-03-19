#!/usr/bin/env python3
"""
Disease-Drug-Target Discovery Example

Demonstrates how to discover therapeutic targets and drugs for diseases.
"""

import requests
import re
from typing import Dict, List, Optional, Tuple


KEGG_API_BASE = "https://rest.kegg.jp"


def kegg_find(database: str, query: str) -> List[Tuple[str, str]]:
    """Search KEGG database for entries matching query."""
    url = f"{KEGG_API_BASE}/find/{database}/{query}"
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
    """Retrieve KEGG entry by ID."""
    # Auto-format ID
    if not any(entry_id.startswith(p) for p in ["ds:", "dr:", "hsa:"]):
        if entry_id.startswith("H"):
            entry_id = f"ds:{entry_id}"
        elif entry_id.startswith("D"):
            entry_id = f"dr:{entry_id}"

    url = f"{KEGG_API_BASE}/get/{entry_id}"
    if option:
        url += f"/{option}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_disease_entry(entry: str) -> Dict:
    """
    Parse KEGG DISEASE entry into structured dictionary.

    Args:
        entry: Raw KEGG disease entry text

    Returns:
        Parsed disease information dictionary
    """
    disease = {
        "id": "",
        "name": "",
        "description": "",
        "category": "",
        "genes": [],
        "pathways": [],
        "drugs": [],
        "networks": [],
        "references": [],
        "external_ids": {},
    }

    lines = entry.strip().split("\n")
    current_field = None
    gene_list = []
    drug_list = []

    for line in lines:
        if not line:
            continue

        field_name = line[:12].strip() if len(line) > 12 else ""
        field_value = line[12:].strip() if len(line) > 12 else line.strip()

        if field_name:
            current_field = field_name

            if field_name == "ENTRY":
                disease["id"] = field_value.split()[0]
            elif field_name == "NAME":
                disease["name"] = field_value.rstrip(";")
            elif field_name == "DESCRIPTION":
                disease["description"] = field_value
            elif field_name == "CATEGORY":
                disease["category"] = field_value
            elif field_name == "GENE":
                gene_list.append(field_value)
            elif field_name == "DRUG":
                drug_list.append(field_value)
            elif field_name == "PATHWAY":
                parts = field_value.split(None, 1)
                if len(parts) == 2:
                    disease["pathways"].append({
                        "id": parts[0].strip("()"),
                        "name": parts[1]
                    })
            elif field_name == "NETWORK":
                disease["networks"].append(field_value)
            elif field_name == "DBLINKS":
                if ":" in field_value:
                    db, val = field_value.split(":", 1)
                    disease["external_ids"][db.strip()] = val.strip()
            elif field_name == "REFERENCE":
                disease["references"].append({"raw": field_value})
        elif current_field in ["GENE", "DRUG", "DESCRIPTION", "REFERENCE"]:
            if current_field == "GENE":
                gene_list.append(field_value)
            elif current_field == "DRUG":
                drug_list.append(field_value)
            elif current_field == "DESCRIPTION":
                disease["description"] += " " + field_value

    # Parse genes
    for gene_str in gene_list:
        parsed = _parse_gene_entry(gene_str)
        if parsed:
            disease["genes"].append(parsed)

    # Parse drugs
    for drug_str in drug_list:
        parsed = _parse_drug_entry(drug_str)
        if parsed:
            disease["drugs"].append(parsed)

    return disease


def _parse_gene_entry(gene_str: str) -> Optional[Dict]:
    """Parse gene entry from disease record."""
    # Format: "(T2D1) CAPN10 [HSA:11132] [KO:K08579]"
    # or: "KCNJ11 [HSA:3767] [KO:K05004]"

    gene = {"raw": gene_str}

    # Extract alias in parentheses
    alias_match = re.match(r"\((\w+)\)\s*", gene_str)
    if alias_match:
        gene["alias"] = alias_match.group(1)
        gene_str = gene_str[alias_match.end():]

    # Extract gene symbol
    symbol_match = re.match(r"(\w+)", gene_str)
    if symbol_match:
        gene["symbol"] = symbol_match.group(1)

    # Extract HSA (Entrez) ID
    hsa_match = re.search(r"\[HSA:(\d+)\]", gene_str)
    if hsa_match:
        gene["entrez_id"] = hsa_match.group(1)

    # Extract KO ID
    ko_match = re.search(r"\[KO:(K\d+)\]", gene_str)
    if ko_match:
        gene["ko"] = ko_match.group(1)

    return gene if gene.get("symbol") else None


def _parse_drug_entry(drug_str: str) -> Optional[Dict]:
    """Parse drug entry from disease record."""
    # Format: "Insulin human [DR:D03230]"
    # or: "Metformin hydrochloride [DR:D00944]"

    drug = {"raw": drug_str}

    # Extract drug ID
    id_match = re.search(r"\[DR:(D\d+)\]", drug_str)
    if id_match:
        drug["id"] = id_match.group(1)
        drug["name"] = drug_str[:id_match.start()].strip()
    else:
        drug["name"] = drug_str.strip()

    return drug


def format_disease_report(disease: Dict) -> str:
    """Format disease information as human-readable report."""
    lines = [
        "=" * 70,
        "KEGG DISEASE REPORT",
        "=" * 70,
        f"Disease ID:   {disease['id']}",
        f"Name:         {disease['name']}",
        f"Category:     {disease['category']}",
        "",
    ]

    if disease["description"]:
        lines.extend([
            "DESCRIPTION",
            "-" * 40,
            disease["description"][:600] + ("..." if len(disease["description"]) > 600 else ""),
            "",
        ])

    lines.extend([
        f"ASSOCIATED GENES ({len(disease['genes'])})",
        "-" * 40,
    ])
    for gene in disease["genes"][:10]:
        symbol = gene.get("symbol", "?")
        alias = gene.get("alias", "")
        ko = gene.get("ko", "")
        prefix = f"({alias}) " if alias else ""
        lines.append(f"  • {prefix}{symbol}" + (f" [KO:{ko}]" if ko else ""))
    if len(disease["genes"]) > 10:
        lines.append(f"  ... and {len(disease['genes']) - 10} more genes")
    lines.append("")

    lines.extend([
        f"APPROVED DRUGS ({len(disease['drugs'])})",
        "-" * 40,
    ])

    # Group drugs by category
    insulins = [d for d in disease["drugs"] if "insulin" in d["name"].lower()]
    metformins = [d for d in disease["drugs"] if "metformin" in d["name"].lower()]
    glp1 = [d for d in disease["drugs"] if any(x in d["name"].lower() for x in ["glutide", "enatide", "zepatide"])]
    sglt2 = [d for d in disease["drugs"] if any(x in d["name"].lower() for x in ["flozin", "gliflozin"])]
    others = [d for d in disease["drugs"] if d not in insulins + metformins + glp1 + sglt2]

    if insulins:
        lines.append("  Insulins:")
        for drug in insulins[:5]:
            lines.append(f"    • {drug['name']} [{drug.get('id', '?')}]")
    if metformins:
        lines.append("  Biguanides:")
        for drug in metformins[:3]:
            lines.append(f"    • {drug['name']} [{drug.get('id', '?')}]")
    if glp1:
        lines.append("  GLP-1 Agonists:")
        for drug in glp1[:5]:
            lines.append(f"    • {drug['name']} [{drug.get('id', '?')}]")
    if sglt2:
        lines.append("  SGLT2 Inhibitors:")
        for drug in sglt2[:5]:
            lines.append(f"    • {drug['name']} [{drug.get('id', '?')}]")
    if others:
        lines.append("  Other Drugs:")
        for drug in others[:5]:
            lines.append(f"    • {drug['name']} [{drug.get('id', '?')}]")
    lines.append("")

    if disease["pathways"]:
        lines.extend([
            "PATHWAYS",
            "-" * 40,
        ])
        for pw in disease["pathways"]:
            lines.append(f"  • {pw['id']}: {pw['name']}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Example: Discover drugs and targets for Type 2 diabetes."""
    print("Searching for 'diabetes' in KEGG DISEASE database...\n")

    # Step 1: Find disease
    results = kegg_find("disease", "diabetes")
    print(f"Found {len(results)} matches:")
    for entry_id, description in results[:10]:
        print(f"  {entry_id}: {description}")
    print()

    # Step 2: Get Type 2 diabetes (H00409)
    print("=" * 70)
    print("Fetching full entry for H00409 (Type 2 diabetes mellitus)...\n")

    entry = kegg_get("H00409")
    disease = parse_disease_entry(entry)

    # Print formatted report
    print(format_disease_report(disease))

    # Summary for drug discovery
    print("\n" + "=" * 70)
    print("DRUG DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Total associated genes:  {len(disease['genes'])}")
    print(f"Total approved drugs:    {len(disease['drugs'])}")
    print(f"Pathway involvement:     {len(disease['pathways'])} pathways")

    # Print KO terms for target identification
    ko_terms = [g["ko"] for g in disease["genes"] if g.get("ko")]
    print(f"\nTarget KO terms: {', '.join(ko_terms[:10])}")

    # Print as JSON for programmatic use
    print("\n" + "=" * 70)
    print("SAMPLE JSON OUTPUT")
    print("=" * 70)
    import json
    sample = {
        "id": disease["id"],
        "name": disease["name"],
        "category": disease["category"],
        "gene_count": len(disease["genes"]),
        "drug_count": len(disease["drugs"]),
        "sample_genes": disease["genes"][:5],
        "sample_drugs": disease["drugs"][:5],
        "pathways": disease["pathways"],
    }
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
