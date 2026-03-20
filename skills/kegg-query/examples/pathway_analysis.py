#!/usr/bin/env python3
"""
Pathway Analysis Example

Demonstrates how to analyze KEGG pathways to retrieve genes, compounds, and modules.
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
    url = f"{KEGG_API_BASE}/get/{entry_id}"
    if option:
        url += f"/{option}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def kegg_link(target_db: str, source_db_or_id: str) -> List[Tuple[str, str]]:
    """Find related entries between databases."""
    url = f"{KEGG_API_BASE}/link/{target_db}/{source_db_or_id}"
    response = requests.get(url)
    response.raise_for_status()

    results = []
    for line in response.text.strip().split("\n"):
        if line:
            parts = line.split("\t")
            if len(parts) == 2:
                results.append((parts[0], parts[1]))
    return results


def parse_pathway_entry(entry: str) -> Dict:
    """
    Parse KEGG PATHWAY entry into structured dictionary.

    Args:
        entry: Raw KEGG pathway entry text

    Returns:
        Parsed pathway information dictionary
    """
    pathway = {
        "id": "",
        "name": "",
        "description": "",
        "class": "",
        "organism": "",
        "modules": [],
        "genes": [],
        "compounds": [],
        "drugs": [],
        "diseases": [],
        "related_pathways": [],
        "references": [],
    }

    lines = entry.strip().split("\n")
    current_field = None
    compound_list = []
    gene_list = []
    module_list = []

    for line in lines:
        if not line:
            continue

        field_name = line[:12].strip() if len(line) > 12 else ""
        field_value = line[12:].strip() if len(line) > 12 else line.strip()

        if field_name:
            current_field = field_name

            if field_name == "ENTRY":
                pathway["id"] = field_value.split()[0]
            elif field_name == "NAME":
                pathway["name"] = field_value
            elif field_name == "DESCRIPTION":
                pathway["description"] = field_value
            elif field_name == "CLASS":
                pathway["class"] = field_value
            elif field_name == "ORGANISM":
                pathway["organism"] = field_value
            elif field_name == "MODULE":
                module_list.append(field_value)
            elif field_name == "GENE":
                gene_list.append(field_value)
            elif field_name == "COMPOUND":
                compound_list.append(field_value)
            elif field_name == "DRUG":
                pathway["drugs"].append(_parse_compound_like(field_value))
            elif field_name == "REL_PATHWAY":
                parts = field_value.split(None, 1)
                if len(parts) == 2:
                    pathway["related_pathways"].append({
                        "id": parts[0],
                        "name": parts[1]
                    })
            elif field_name == "REFERENCE":
                pathway["references"].append({"raw": field_value})
        elif current_field in ["GENE", "COMPOUND", "MODULE", "REFERENCE"]:
            if current_field == "GENE":
                gene_list.append(field_value)
            elif current_field == "COMPOUND":
                compound_list.append(field_value)
            elif current_field == "MODULE":
                module_list.append(field_value)

    # Parse genes
    for gene_str in gene_list:
        parsed = _parse_gene_entry(gene_str)
        if parsed:
            pathway["genes"].append(parsed)

    # Parse compounds
    for cpd_str in compound_list:
        parsed = _parse_compound_like(cpd_str)
        if parsed:
            pathway["compounds"].append(parsed)

    # Parse modules
    for mod_str in module_list:
        parts = mod_str.split(None, 1)
        if len(parts) == 2:
            pathway["modules"].append({
                "id": parts[0],
                "description": parts[1].replace("[PATH:" + pathway["id"] + "]", "").strip()
            })

    return pathway


def _parse_gene_entry(gene_str: str) -> Optional[Dict]:
    """Parse gene entry string."""
    # Format: "10327  AKR1A1; aldo-keto reductase family 1 member A1 [KO:K00002] [EC:1.1.1.2]"
    match = re.match(r"(\d+)\s+(\w+);\s*(.+?)(?:\s+\[KO:(K\d+)\])?(?:\s+\[EC:([^\]]+)\])?$", gene_str)
    if match:
        return {
            "id": match.group(1),
            "symbol": match.group(2),
            "name": match.group(3).strip(),
            "ko": match.group(4) or "",
            "ec": match.group(5) or "",
        }
    # Simpler format
    parts = gene_str.split(None, 1)
    if len(parts) >= 2:
        return {"id": parts[0], "raw": gene_str}
    return None


def _parse_compound_like(cpd_str: str) -> Optional[Dict]:
    """Parse compound/drug entry string."""
    # Format: "C00022  Pyruvate"
    parts = cpd_str.split(None, 1)
    if len(parts) == 2:
        return {"id": parts[0], "name": parts[1]}
    return None


def format_pathway_report(pathway: Dict) -> str:
    """Format pathway information as human-readable report."""
    lines = [
        "=" * 70,
        "KEGG PATHWAY REPORT",
        "=" * 70,
        f"Pathway ID:   {pathway['id']}",
        f"Name:         {pathway['name']}",
        f"Class:        {pathway['class']}",
        f"Organism:     {pathway['organism']}",
        "",
    ]

    if pathway["description"]:
        lines.extend([
            "DESCRIPTION",
            "-" * 40,
            pathway["description"][:500] + ("..." if len(pathway["description"]) > 500 else ""),
            "",
        ])

    if pathway["modules"]:
        lines.extend([
            f"MODULES ({len(pathway['modules'])})",
            "-" * 40,
        ])
        for mod in pathway["modules"][:5]:
            lines.append(f"  • {mod['id']}: {mod['description'][:50]}")
        if len(pathway["modules"]) > 5:
            lines.append(f"  ... and {len(pathway['modules']) - 5} more")
        lines.append("")

    lines.extend([
        f"GENES ({len(pathway['genes'])})",
        "-" * 40,
    ])
    for gene in pathway["genes"][:10]:
        symbol = gene.get("symbol", gene.get("id", "?"))
        ko = gene.get("ko", "")
        name = gene.get("name", "")[:30]
        lines.append(f"  • {symbol}" + (f" [KO:{ko}]" if ko else "") + f" - {name}")
    if len(pathway["genes"]) > 10:
        lines.append(f"  ... and {len(pathway['genes']) - 10} more genes")
    lines.append("")

    lines.extend([
        f"COMPOUNDS ({len(pathway['compounds'])})",
        "-" * 40,
    ])
    for cpd in pathway["compounds"][:10]:
        lines.append(f"  • {cpd['id']}: {cpd['name']}")
    if len(pathway["compounds"]) > 10:
        lines.append(f"  ... and {len(pathway['compounds']) - 10} more compounds")
    lines.append("")

    if pathway["related_pathways"]:
        lines.extend([
            "RELATED PATHWAYS",
            "-" * 40,
        ])
        for rp in pathway["related_pathways"]:
            lines.append(f"  • {rp['id']}: {rp['name']}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Example: Analyze Glycolysis pathway (hsa00010)."""
    print("Analyzing KEGG pathway: hsa00010 (Glycolysis/Gluconeogenesis)\n")

    # Get pathway entry
    entry = kegg_get("hsa00010")
    pathway = parse_pathway_entry(entry)

    # Print formatted report
    print(format_pathway_report(pathway))

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total genes:      {len(pathway['genes'])}")
    print(f"Total compounds:  {len(pathway['compounds'])}")
    print(f"Total modules:    {len(pathway['modules'])}")

    # Print unique KO terms (orthology)
    ko_terms = set()
    for gene in pathway["genes"]:
        if gene.get("ko"):
            ko_terms.add(gene["ko"])
    print(f"Unique KO terms:  {len(ko_terms)}")

    # Print as JSON for programmatic use
    print("\n" + "=" * 70)
    print("SAMPLE JSON OUTPUT")
    print("=" * 70)
    import json
    sample = {
        "id": pathway["id"],
        "name": pathway["name"],
        "gene_count": len(pathway["genes"]),
        "compound_count": len(pathway["compounds"]),
        "sample_genes": pathway["genes"][:3],
        "sample_compounds": pathway["compounds"][:5],
    }
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
