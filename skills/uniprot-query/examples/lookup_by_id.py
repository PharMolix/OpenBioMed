"""
UniProt Query - Use Case 1: Protein Lookup by ID

Fetch comprehensive protein information including metadata and generate reports.

Usage:
    python lookup_by_id.py P0DTC2
    python lookup_by_id.py P00533 --output ./output
"""

import argparse
import json
import os
import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple

# OpenBioMed imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.data import Protein


def parse_uniprot_entry(data: Dict) -> Dict[str, Any]:
    """Parse UniProt JSON entry into structured metadata."""
    metadata = {
        'accession': data.get('primaryAccession'),
        'uniProtId': data.get('uniProtkbId'),
        'entryType': data.get('entryType'),
        'protein': {},
        'gene': {},
        'organism': {},
        'sequence': {},
        'function': [],
        'diseases': [],
        'keywords': [],
        'domains': [],
        'subcellular_location': [],
        'ptms': [],
        'cross_references': {},
        'query_date': datetime.now().isoformat()
    }

    # Protein name
    prot_desc = data.get('proteinDescription', {})
    rec_name = prot_desc.get('recommendedName', {})
    if rec_name:
        metadata['protein']['name'] = rec_name.get('fullName', {}).get('value')
        ec_nums = rec_name.get('ecNumbers', [])
        metadata['protein']['ec_numbers'] = [ec.get('value') for ec in ec_nums]

    alt_names = prot_desc.get('alternativeNames', [])
    metadata['protein']['alternative_names'] = [
        alt.get('fullName', {}).get('value') for alt in alt_names if alt.get('fullName')
    ]

    # Gene name
    genes = data.get('genes', [])
    if genes:
        metadata['gene']['primary'] = genes[0].get('geneName', {}).get('value')
        metadata['gene']['synonyms'] = [s.get('value') for s in genes[0].get('synonyms', [])]

    # Organism
    org = data.get('organism', {})
    metadata['organism']['scientific_name'] = org.get('scientificName')
    metadata['organism']['common_name'] = org.get('commonName')
    metadata['organism']['taxon_id'] = org.get('taxonId')

    # Sequence
    seq_info = data.get('sequence', {})
    metadata['sequence']['length'] = seq_info.get('length')
    metadata['sequence']['mass'] = seq_info.get('molWeight')
    metadata['sequence']['value'] = seq_info.get('value')

    # Comments (function, disease, location)
    comments = data.get('comments', [])
    for comment in comments:
        comment_type = comment.get('commentType')

        if comment_type == 'FUNCTION':
            for text in comment.get('texts', []):
                metadata['function'].append(text.get('value'))

        elif comment_type == 'DISEASE':
            disease = comment.get('disease', {})
            metadata['diseases'].append({
                'id': disease.get('diseaseId'),
                'description': disease.get('description'),
                'acronym': disease.get('acronym'),
                'cross_reference': disease.get('diseaseCrossReference', {})
            })

        elif comment_type == 'SUBCELLULAR LOCATION':
            for loc in comment.get('subcellularLocations', []):
                loc_value = loc.get('location', {}).get('value')
                if loc_value:
                    metadata['subcellular_location'].append(loc_value)

        elif comment_type == 'PTM':
            for text in comment.get('texts', []):
                metadata['ptms'].append(text.get('value'))

    # Keywords
    keywords = data.get('keywords', [])
    metadata['keywords'] = [kw.get('name') for kw in keywords]

    # Features (domains, regions, motifs)
    features = data.get('features', [])
    for feat in features:
        feat_type = feat.get('type')
        if feat_type in ['Domain', 'Region', 'Repeat', 'Motif', 'Active site', 'Binding site']:
            loc = feat.get('location', {})
            metadata['domains'].append({
                'type': feat_type,
                'description': feat.get('description'),
                'start': loc.get('start', {}).get('value'),
                'end': loc.get('end', {}).get('value')
            })

    # Cross-references
    xrefs = data.get('uniProtKBCrossReferences', [])
    for xref in xrefs[:20]:  # Limit to first 20
        db = xref.get('database')
        if db not in metadata['cross_references']:
            metadata['cross_references'][db] = []
        metadata['cross_references'][db].append(xref.get('id'))

    return metadata


def generate_report(metadata: Dict) -> str:
    """Generate human-readable text report from metadata."""
    lines = []
    lines.append('=' * 70)
    lines.append('UNIPROT PROTEIN REPORT')
    lines.append('=' * 70)
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')

    # Identification
    lines.append('─' * 70)
    lines.append('IDENTIFICATION')
    lines.append('─' * 70)
    lines.append(f'Accession:     {metadata.get("accession", "N/A")}')
    lines.append(f'UniProt ID:    {metadata.get("uniProtId", "N/A")}')
    lines.append(f'Entry Type:    {metadata.get("entryType", "N/A")}')
    lines.append('')

    # Protein
    lines.append('─' * 70)
    lines.append('PROTEIN')
    lines.append('─' * 70)
    prot = metadata.get('protein', {})
    lines.append(f'Name:          {prot.get("name", "N/A")}')
    if prot.get('ec_numbers'):
        lines.append(f'EC Numbers:    {", ".join(prot["ec_numbers"])}')
    if prot.get('alternative_names'):
        lines.append(f'Alt Names:     {", ".join(prot["alternative_names"][:3])}')
    lines.append('')

    # Gene
    lines.append('─' * 70)
    lines.append('GENE')
    lines.append('─' * 70)
    gene = metadata.get('gene', {})
    lines.append(f'Primary:       {gene.get("primary", "N/A")}')
    if gene.get('synonyms'):
        lines.append(f'Synonyms:      {", ".join(gene["synonyms"])}')
    lines.append('')

    # Organism
    lines.append('─' * 70)
    lines.append('ORGANISM')
    lines.append('─' * 70)
    org = metadata.get('organism', {})
    lines.append(f'Scientific:    {org.get("scientific_name", "N/A")}')
    lines.append(f'Common:        {org.get("common_name", "N/A")}')
    lines.append(f'Taxon ID:      {org.get("taxon_id", "N/A")}')
    lines.append('')

    # Sequence
    lines.append('─' * 70)
    lines.append('SEQUENCE')
    lines.append('─' * 70)
    seq = metadata.get('sequence', {})
    lines.append(f'Length:        {seq.get("length", "N/A")} amino acids')
    lines.append(f'Mass:          {seq.get("mass", "N/A")} Da')
    lines.append('')

    # Function
    lines.append('─' * 70)
    lines.append('FUNCTION')
    lines.append('─' * 70)
    for i, func in enumerate(metadata.get('function', [])[:3], 1):
        func_text = func[:300] + '...' if len(func) > 300 else func
        lines.append(f'{i}. {func_text}')
        lines.append('')

    # Diseases
    lines.append('─' * 70)
    lines.append('DISEASES')
    lines.append('─' * 70)
    diseases = metadata.get('diseases', [])
    if diseases:
        for disease in diseases:
            lines.append(f'• {disease.get("id", "N/A")} ({disease.get("acronym", "N/A")})')
            desc = disease.get('description', 'No description')
            lines.append(f'  {desc[:150]}...' if len(desc) > 150 else f'  {desc}')
    else:
        lines.append('No disease annotations')
    lines.append('')

    # Subcellular Location
    lines.append('─' * 70)
    lines.append('SUBCELLULAR LOCATION')
    lines.append('─' * 70)
    locations = metadata.get('subcellular_location', [])
    if locations:
        for loc in locations:
            lines.append(f'• {loc}')
    else:
        lines.append('No location annotations')
    lines.append('')

    # Keywords
    lines.append('─' * 70)
    lines.append('KEYWORDS')
    lines.append('─' * 70)
    keywords = metadata.get('keywords', [])
    lines.append(', '.join(keywords[:15]))
    lines.append('')

    # Domains
    lines.append('─' * 70)
    lines.append('DOMAINS/FEATURES')
    lines.append('─' * 70)
    domains = metadata.get('domains', [])
    for domain in domains[:10]:
        start = domain.get('start', '?')
        end = domain.get('end', '?')
        lines.append(f'• {domain.get("type")}: {domain.get("description")} ({start}-{end})')
    lines.append('')

    # Cross-references
    lines.append('─' * 70)
    lines.append('CROSS-REFERENCES')
    lines.append('─' * 70)
    xrefs = metadata.get('cross_references', {})
    for db, ids in list(xrefs.items())[:10]:
        lines.append(f'{db}: {", ".join(ids[:5])}')
    lines.append('')

    lines.append('=' * 70)
    lines.append('END OF REPORT')
    lines.append('=' * 70)

    return '\n'.join(lines)


def lookup_protein(accession: str, output_dir: str = './tmp/uniprot') -> Tuple[Dict, str, Protein]:
    """
    Look up protein by UniProt accession.

    Args:
        accession: UniProt accession (e.g., P00533, P0DTC2)
        output_dir: Directory for output files

    Returns:
        Tuple of (metadata dict, report text, Protein object)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get protein sequence using OpenBioMed tool
    print(f"Fetching protein sequence for {accession}...")
    tool = TOOLS["protein_uniprot_request"]
    proteins, message = tool.run(accession=accession)
    protein = proteins[0]
    print(f"  Sequence length: {len(protein.sequence)} aa")

    # Step 2: Fetch full metadata from UniProt REST API
    print(f"Fetching metadata from UniProt API...")
    url = f"https://rest.uniprot.org/uniprotkb/{accession}?format=json"
    response = requests.get(url)
    response.raise_for_status()

    # Step 3: Parse metadata
    metadata = parse_uniprot_entry(response.json())
    print(f"  Protein: {metadata['protein'].get('name', 'N/A')}")
    print(f"  Gene: {metadata['gene'].get('primary', 'N/A')}")

    # Step 4: Generate report
    report = generate_report(metadata)

    # Step 5: Save outputs
    metadata_file = os.path.join(output_dir, f"{accession}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved: {metadata_file}")

    report_file = os.path.join(output_dir, f"{accession}_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_file}")

    return metadata, report, protein


def main():
    parser = argparse.ArgumentParser(description='Lookup protein by UniProt accession')
    parser.add_argument('accession', help='UniProt accession (e.g., P00533, P0DTC2)')
    parser.add_argument('--output', '-o', default='./tmp/uniprot', help='Output directory')
    args = parser.parse_args()

    metadata, report, protein = lookup_protein(args.accession, args.output)

    print("\n" + report)


if __name__ == "__main__":
    main()
