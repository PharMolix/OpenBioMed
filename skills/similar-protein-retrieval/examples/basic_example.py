"""
Similar Protein Retrieval - Basic Example

This script demonstrates how to retrieve similar proteins using
UniProt, PDB, FASTA, or PDB file inputs with FoldSeek or MSA.
"""

import os
import sys
import asyncio
import aiohttp

# Add OpenBioMed to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import requests
import pandas as pd
import glob

from open_biomed.data import Protein
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.tools.web_request_tools import MSARequester, FoldSeekRequester


def has_loaded_structure_check(protein: Protein) -> bool:
    """Check if protein has actual 3D structure loaded (not just sequence)."""
    return hasattr(protein, 'residues') and protein.residues is not None


async def parse_input_async(user_input: str):
    """
    Parse input and return Protein object with structure info.

    Args:
        user_input: UniProt ID, PDB ID, FASTA string, or file path

    Returns:
        tuple: (Protein object, has_structure: bool, input_type: str, extra_info: dict)
    """
    extra_info = {}

    # Check if it's a file path
    if os.path.isfile(user_input):
        if user_input.endswith('.pdb'):
            protein = Protein.from_pdb_file(user_input)
            return protein, True, "pdb_file", extra_info
        elif user_input.endswith(('.fasta', '.fa')):
            with open(user_input) as f:
                lines = f.readlines()
            seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
            protein = Protein.from_fasta(seq)
            return protein, False, "fasta_file", extra_info

    # Check if it's a UniProt ID (e.g., P0DTC2)
    if len(user_input) in [6, 10] and user_input[0].isalpha() and user_input[1].isdigit():
        return query_uniprot(user_input)

    # Check if it's a PDB ID (4 characters, e.g., 6LZG)
    if len(user_input) == 4 and user_input[0].isdigit():
        return await query_pdb_async(user_input)

    # Assume it's a FASTA sequence
    protein = Protein.from_fasta(user_input)
    return protein, False, "fasta_string", extra_info


def has_loaded_structure(protein: Protein) -> bool:
    """Check if protein has actual 3D structure loaded (not just sequence)."""
    return hasattr(protein, 'residues') and protein.residues is not None


def query_uniprot(uniprot_id: str):
    """
    Query UniProt for sequence and PDB cross-references.

    Args:
        uniprot_id: UniProt accession (e.g., P0DTC2)

    Returns:
        tuple: (Protein object, has_structure: bool, input_type: str, extra_info: dict)
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Extract sequence
    sequence = data['sequence']['value']
    protein = Protein.from_fasta(sequence)
    protein.name = uniprot_id

    # Extract metadata
    extra_info = {
        "protein_name": data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
        "organism": data.get('organism', {}).get('scientificName', 'Unknown'),
        "gene": data.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown') if data.get('genes') else 'Unknown',
    }

    # Get PDB cross-references
    xrefs = data.get('uniProtKBCrossReferences', [])
    pdb_entries = [x for x in xrefs if x.get('database') == 'PDB']

    # Parse PDB info
    extra_info["pdb_refs"] = []
    for entry in pdb_entries[:10]:  # Limit to first 10
        properties = {p.get('key'): p.get('value') for p in entry.get('properties', [])}
        extra_info["pdb_refs"].append({
            "pdb_id": entry.get('id'),
            "method": properties.get('Method', 'N/A'),
            "resolution": properties.get('Resolution', 'N/A'),
            "chains": properties.get('Chains', 'N/A')
        })

    has_structure = len(pdb_entries) > 0
    return protein, has_structure, "uniprot", extra_info


async def query_pdb_async(pdb_id: str):
    """
    Download PDB file and load structure.

    Args:
        pdb_id: 4-character PDB ID (e.g., 6LZG)

    Returns:
        tuple: (Protein object, has_structure: bool, input_type: str, extra_info: dict)
    """
    # Download PDB file directly
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.text()
            else:
                raise ValueError(f"PDB ID {pdb_id} not found (status {response.status})")

    # Save to temp file (synchronous write is fine)
    pdb_file = f"./tmp/pdb_{pdb_id}.pdb"
    os.makedirs("./tmp", exist_ok=True)
    with open(pdb_file, 'w') as f:
        f.write(content)

    protein = Protein.from_pdb_file(pdb_file)
    protein.name = pdb_id

    extra_info = {"pdb_file": pdb_file}
    return protein, True, "pdb_id", extra_info


async def run_msa(protein: Protein):
    """
    Run MSA for sequence similarity search.

    Args:
        protein: OpenBioMed Protein object

    Returns:
        str: Path to .a3m file with MSA results
    """
    print("Running MSA (sequence similarity search)...")
    msa = MSARequester()
    result, _ = await msa.run_async(protein)
    print(f"MSA results saved to: {result[0]}")
    return result[0]


async def run_foldseek(protein: Protein, databases=None, timeout=120):
    """
    Run FoldSeek for structure similarity search.

    Args:
        protein: OpenBioMed Protein object
        databases: List of databases to search (default: pdb100, afdb50)
        timeout: Timeout in seconds (default: 120)

    Returns:
        str: Path to results directory
    """
    if databases is None:
        databases = ["pdb100"]

    print("Running FoldSeek (structure similarity search)...")
    print("Note: This may take 30 seconds to 2 minutes depending on protein size.")
    print("      If it times out, the FoldSeek server may be busy. Try again later.")

    try:
        foldseek = FoldSeekRequester(database=databases, timeout=timeout)
        result, _ = await foldseek.run_async(protein)
        print(f"FoldSeek results saved to: {result[0]}")
        return result[0]
    except asyncio.TimeoutError:
        print("\nERROR: FoldSeek request timed out.")
        print("The server may be busy. You can:")
        print("  1. Try again with a smaller protein/domain")
        print("  2. Use the web interface: https://search.foldseek.com")
        print("  3. Try MSA instead (sequence-based search)")
        return None
    except Exception as e:
        print(f"\nERROR: FoldSeek failed: {e}")
        return None


def parse_foldseek_results(result_dir: str, top_n: int = 20):
    """
    Parse FoldSeek .m8 output file.

    Args:
        result_dir: Path to FoldSeek results directory
        top_n: Number of top results to return

    Returns:
        list: List of dicts with target, identity, alignment_length, evalue
    """
    m8_files = glob.glob(f"{result_dir}/*.m8")

    if not m8_files:
        print(f"No .m8 files found in {result_dir}")
        return []

    # Use the main results file (not the report file)
    m8_file = [f for f in m8_files if 'report' not in f][0] if len(m8_files) > 1 else m8_files[0]

    df = pd.read_csv(m8_file, sep='\t', header=None)

    results = []
    for _, row in df.head(top_n).iterrows():
        results.append({
            "target": row[1][:60] + "..." if len(str(row[1])) > 60 else row[1],
            "identity": f"{row[2]:.1f}%",
            "alignment_length": int(row[3]),
            "evalue": f"{row[11]:.2e}" if len(df.columns) > 11 else "N/A"
        })

    return results


def display_results(results: list, search_type: str):
    """Display results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"Top {len(results)} similar proteins ({search_type})")
    print('=' * 80)
    print(f"{'Target':<65} | {'Identity':>8} | {'E-value':>10}")
    print('-' * 80)
    for r in results:
        print(f"{r['target']:<65} | {r['identity']:>8} | {r['evalue']:>10}")


async def main():
    """Main workflow example."""
    print("=" * 80)
    print("Similar Protein Retrieval - Example")
    print("=" * 80)

    # Example inputs to test (uncomment one)
    # user_input = "P0DTC2"  # UniProt ID (SARS-CoV-2 Spike)
    user_input = "6LZG"    # PDB ID (SARS-CoV-2 RBD-ACE2 complex)
    # user_input = "P0DTC2"  # Default example

    print(f"\nInput: {user_input}")

    # Step 1: Parse input
    print("\n--- Step 1: Parsing input ---")
    protein, has_structure_ref, input_type, extra_info = await parse_input_async(user_input)

    # Check if actual 3D structure is loaded (not just PDB references)
    has_loaded_structure = has_loaded_structure_check(protein)

    print(f"Input type: {input_type}")
    print(f"Sequence length: {len(protein.sequence)} aa")
    print(f"Has 3D structure loaded: {has_loaded_structure}")

    if input_type == "uniprot":
        print(f"Protein name: {extra_info.get('protein_name', 'N/A')}")
        print(f"Organism: {extra_info.get('organism', 'N/A')}")
        print(f"PDB structures referenced: {len(extra_info.get('pdb_refs', []))}")

    # Step 2: Choose search method
    print("\n--- Step 2: Choosing search method ---")

    if not has_loaded_structure:
        search_method = "msa"
        print("No 3D structure loaded. Using MSA (sequence similarity).")
        if input_type == "uniprot" and extra_info.get('pdb_refs'):
            print(f"Note: {len(extra_info['pdb_refs'])} PDB structures are referenced.")
            print("      To use FoldSeek, provide a PDB ID from the references.")
    else:
        print("3D structure loaded!")
        print("Options:")
        print("  1. MSA - Sequence similarity (searches UniRef database)")
        print("  2. FoldSeek - Structure similarity (searches PDB/AFDB)")
        # For automated example, default to FoldSeek
        search_method = "foldseek"
        print(f"Defaulting to: FoldSeek (structure similarity)")

    # Step 3: Run similarity search
    print("\n--- Step 3: Running similarity search ---")

    if search_method == "msa":
        result_path = await run_msa(protein)
        print(f"\nMSA results saved to: {result_path}")
        print("Note: MSA produces .a3m file with aligned sequences.")
        print("Use external tools (e.g., HHblits, BLAST) to extract specific hits.")
    else:
        result_path = await run_foldseek(protein)
        if result_path:
            results = parse_foldseek_results(result_path, top_n=15)
            display_results(results, "FoldSeek - Structure Similarity")
        else:
            print("FoldSeek search failed. Try using MSA instead.")

    print("\n" + "=" * 80)
    print("Workflow complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
