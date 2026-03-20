"""
UniProt Query - Use Case 2: Search by Criteria

Search UniProt database by gene name, organism, keywords, or disease.

Usage:
    python search_by_criteria.py "gene_exact:EGFR AND organism_id:9606"
    python search_by_criteria.py "keyword:Kinase" --organism 9606 --limit 20
    python search_by_criteria.py "diabetes" --organism 9606 --reviewed
"""

import argparse
import json
import os
import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# Common organism IDs
ORGANISM_IDS = {
    'human': 9606,
    'mouse': 10090,
    'rat': 10116,
    'ecoli': 83333,
    'yeast': 559292,
    'sars-cov-2': 2697049,
    'sars-cov': 694009,
}


def build_query(
    query: str,
    organism: Optional[int] = None,
    reviewed: bool = False,
) -> str:
    """Build UniProt query string from components."""
    parts = [query]

    if organism:
        parts.append(f"organism_id:{organism}")

    if reviewed:
        parts.append("reviewed:true")

    return " AND ".join(parts)


def search_uniprot(
    query: str,
    fields: str = "accession,gene_primary,gene_synonym,protein_name,organism_name,length,mass,cc_function,keyword",
    size: int = 10,
    format: str = "json"
) -> Tuple[List[Dict], Dict]:
    """
    Search UniProt by query string.

    Args:
        query: UniProt query string
        fields: Comma-separated list of fields to return
        size: Maximum number of results
        format: Response format (json, tsv, etc.)

    Returns:
        Tuple of (results list, response headers dict)
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    params = {
        'query': query,
        'fields': fields,
        'format': format,
        'size': size
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    data = response.json()
    results = data.get('results', [])

    # Get total count from headers
    headers = {
        'total_results': response.headers.get('x-total-results', 'unknown'),
        'link': response.headers.get('link', '')
    }

    return results, headers


def parse_search_results(results: List[Dict]) -> List[Dict[str, Any]]:
    """Parse search results into structured format."""
    parsed = []

    for entry in results:
        entry_data = {
            'accession': entry.get('primaryAccession'),
            'uniProtId': entry.get('uniProtkbId'),
            'reviewed': 'reviewed' in entry.get('entryType', '').lower(),
        }

        # Gene
        genes = entry.get('genes', [])
        if genes:
            entry_data['gene'] = {
                'primary': genes[0].get('geneName', {}).get('value'),
                'synonyms': [s.get('value') for s in genes[0].get('synonyms', [])]
            }
        else:
            entry_data['gene'] = {'primary': None, 'synonyms': []}

        # Protein
        prot_desc = entry.get('proteinDescription', {})
        rec_name = prot_desc.get('recommendedName', {})
        entry_data['protein_name'] = rec_name.get('fullName', {}).get('value') if rec_name else None

        # Organism
        org = entry.get('organism', {})
        entry_data['organism'] = {
            'scientific_name': org.get('scientificName'),
            'taxon_id': org.get('taxonId')
        }

        # Sequence
        seq = entry.get('sequence', {})
        entry_data['sequence'] = {
            'length': seq.get('length'),
            'mass': seq.get('molWeight')
        }

        # Function
        comments = entry.get('comments', [])
        functions = []
        for comment in comments:
            if comment.get('commentType') == 'FUNCTION':
                for text in comment.get('texts', []):
                    functions.append(text.get('value'))
        entry_data['function'] = functions

        # Keywords
        keywords = entry.get('keywords', [])
        entry_data['keywords'] = [kw.get('name') for kw in keywords]

        parsed.append(entry_data)

    return parsed


def generate_search_report(query: str, results: List[Dict], total: str) -> str:
    """Generate human-readable search results report."""
    lines = []
    lines.append('=' * 70)
    lines.append('UNIPROT SEARCH RESULTS REPORT')
    lines.append('=' * 70)
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Query: {query}')
    lines.append(f'Total Results: {total}')
    lines.append(f'Returned: {len(results)}')
    lines.append('')

    for i, entry in enumerate(results, 1):
        lines.append('─' * 70)
        lines.append(f'ENTRY {i}: {entry.get("accession", "N/A")}')
        lines.append('─' * 70)

        gene = entry.get('gene', {})
        lines.append(f'Gene:          {gene.get("primary", "N/A")}')
        if gene.get('synonyms'):
            lines.append(f'Synonyms:      {", ".join(gene["synonyms"])}')

        lines.append(f'Protein:       {entry.get("protein_name", "N/A")}')

        org = entry.get('organism', {})
        lines.append(f'Organism:      {org.get("scientific_name", "N/A")}')

        seq = entry.get('sequence', {})
        lines.append(f'Length:        {seq.get("length", "N/A")} aa')
        if seq.get('mass'):
            lines.append(f'Mass:          {seq.get("mass")} Da')

        # Function summary
        if entry.get('function'):
            func = entry['function'][0]
            func_short = func[:150] + '...' if len(func) > 150 else func
            lines.append(f'Function:      {func_short}')

        # Keywords
        if entry.get('keywords'):
            lines.append(f'Keywords:      {", ".join(entry["keywords"][:5])}')

        lines.append('')

    lines.append('=' * 70)
    lines.append('END OF REPORT')
    lines.append('=' * 70)

    return '\n'.join(lines)


def search_proteins(
    query: str,
    organism: Optional[int] = None,
    reviewed: bool = False,
    limit: int = 10,
    output_dir: str = './tmp/uniprot'
) -> Tuple[List[Dict], str]:
    """
    Search proteins by criteria.

    Args:
        query: Search query (gene name, keyword, etc.)
        organism: Organism taxon ID (optional)
        reviewed: Only reviewed (Swiss-Prot) entries
        limit: Maximum results to return
        output_dir: Directory for output files

    Returns:
        Tuple of (parsed results list, report text)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build query
    full_query = build_query(query, organism, reviewed)
    print(f"Query: {full_query}")

    # Execute search
    print(f"Searching UniProt...")
    raw_results, headers = search_uniprot(full_query, size=limit)
    total = headers.get('total_results', str(len(raw_results)))
    print(f"  Found: {total} results")

    # Parse results
    parsed_results = parse_search_results(raw_results)

    # Generate report
    report = generate_search_report(full_query, parsed_results, total)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        'query': full_query,
        'query_date': datetime.now().isoformat(),
        'total_results': total,
        'returned_results': len(parsed_results),
        'results': parsed_results
    }

    metadata_file = os.path.join(output_dir, f"search_results_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved: {metadata_file}")

    report_file = os.path.join(output_dir, f"search_results_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_file}")

    return parsed_results, report


def main():
    parser = argparse.ArgumentParser(description='Search UniProt by criteria')
    parser.add_argument('query', help='Search query (gene name, keyword, etc.)')
    parser.add_argument('--organism', '-o', type=int, help='Organism taxon ID (e.g., 9606 for human)')
    parser.add_argument('--organism-name', choices=list(ORGANISM_IDS.keys()),
                        help='Organism name shortcut (human, mouse, etc.)')
    parser.add_argument('--reviewed', '-r', action='store_true',
                        help='Only reviewed (Swiss-Prot) entries')
    parser.add_argument('--limit', '-l', type=int, default=10,
                        help='Maximum results to return (default: 10)')
    parser.add_argument('--output', dest='output_dir', default='./tmp/uniprot',
                        help='Output directory')
    args = parser.parse_args()

    # Resolve organism
    organism = args.organism
    if args.organism_name:
        organism = ORGANISM_IDS[args.organism_name]

    results, report = search_proteins(
        query=args.query,
        organism=organism,
        reviewed=args.reviewed,
        limit=args.limit,
        output_dir=args.output_dir
    )

    print("\n" + report)


if __name__ == "__main__":
    main()
