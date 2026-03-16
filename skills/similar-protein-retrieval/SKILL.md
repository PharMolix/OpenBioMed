---
name: similar-protein-retrieval
description: >
  Retrieve proteins with similar structures, sequences, or from the same family.
  Use this skill when:
  (1) Finding similar proteins or homologs,
  (2) Searching for proteins with similar 3D structure,
  (3) Performing sequence similarity search,
  (4) Discovering proteins in the same family.
license: MIT
category: data-retrieval
tags: [protein-similarity, foldseek, homolog-search, structure-search]
---

# Similar Protein Retrieval

Retrieve proteins with similar structures, sequences, or from the same family using FoldSeek (structure) or MSA (sequence).

## When to Use

- User provides a protein and wants to find similar proteins
- User asks for homologs or orthologs of a protein
- User wants proteins with similar 3D structure
- User wants to search by sequence similarity
- User provides UniProt ID, PDB ID, FASTA, or PDB file as input

## Workflow

### Step 1: Parse Input and Load Protein

Detect input type and load the protein appropriately.

```python
import os
import requests
from open_biomed.data import Protein
from open_biomed.tools.tool_registry import TOOLS

def parse_input(user_input):
    """Parse input and return Protein object with structure info."""
    # Check if it's a file path
    if os.path.isfile(user_input):
        if user_input.endswith('.pdb'):
            return Protein.from_pdb_file(user_input), True, "pdb_file"
        elif user_input.endswith(('.fasta', '.fa')):
            with open(user_input) as f:
                seq = ''.join(l.strip() for l in f if not l.startswith('>'))
            return Protein.from_fasta(seq), False, "fasta_file"

    # Check if it's a UniProt ID (e.g., P0DTC2)
    if len(user_input) in [6, 10] and user_input[0].isalpha():
        return query_uniprot(user_input)

    # Check if it's a PDB ID (4 characters, e.g., 6LZG)
    if len(user_input) == 4 and user_input[0].isdigit():
        return query_pdb(user_input)

    # Assume it's a FASTA sequence
    return Protein.from_fasta(user_input), False, "fasta_string"
```

### Step 2a: Query UniProt (if UniProt ID)

```python
def query_uniprot(uniprot_id):
    """Query UniProt for sequence and PDB cross-references."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    response = requests.get(url)
    data = response.json()

    sequence = data['sequence']['value']
    protein = Protein.from_fasta(sequence)
    protein.name = uniprot_id

    # Get PDB cross-references
    xrefs = data.get('uniProtKBCrossReferences', [])
    pdb_refs = [x['id'] for x in xrefs if x['database'] == 'PDB']

    has_structure = len(pdb_refs) > 0
    return protein, has_structure, "uniprot", {"pdb_refs": pdb_refs}
```

### Step 2b: Query PDB (if PDB ID)

```python
def query_pdb(pdb_id):
    """Download PDB file and load structure."""
    tool = TOOLS["protein_pdb_request"]
    result, _ = tool.run(accession=pdb_id, mode="file_only")
    pdb_file = result[0]
    protein = Protein.from_pdb_file(pdb_file)
    return protein, True, "pdb_id"
```

### Step 3: Choose Similarity Search Method

If 3D structure is available, ask user to choose:

```python
def choose_search_method(has_structure, protein, extra_info=None):
    if not has_structure:
        return "msa"  # Default to MSA for sequence-only input

    print("3D structure available. Choose similarity search method:")
    print("  1. MSA - Sequence similarity (searches UniRef database)")
    print("  2. FoldSeek - Structure similarity (searches PDB/AFDB)")

    choice = input("Enter choice (1 or 2): ").strip()
    return "msa" if choice == "1" else "foldseek"
```

### Step 4a: Run MSA (Sequence Similarity)

```python
from open_biomed.tools.web_request_tools import MSARequester
import asyncio

async def run_msa(protein):
    msa = MSARequester()
    result, _ = await msa.run_async(protein)
    return result[0]  # Path to .a3m file with MSA results
```

### Step 4b: Run FoldSeek (Structure Similarity)

```python
from open_biomed.tools.web_request_tools import FoldSeekRequester
import asyncio

async def run_foldseek(protein):
    foldseek = FoldSeekRequester(database=["pdb100", "afdb50"])
    result, _ = await foldseek.run_async(protein)
    return result[0]  # Path to results directory
```

### Step 5: Parse and Display Results

```python
import pandas as pd
import glob

def parse_foldseek_results(result_dir):
    """Parse FoldSeek .m8 output file."""
    m8_file = glob.glob(f"{result_dir}/*.m8")[0]
    df = pd.read_csv(m8_file, sep='\t', header=None)

    # Columns: query, target, identity, aln_len, mismatch, gap,
    #          q_start, q_end, t_start, t_end, prob, evalue, ...
    results = []
    for _, row in df.iterrows():
        results.append({
            "target": row[1],
            "identity": row[2],
            "alignment_length": row[3],
            "evalue": row[11] if len(df.columns) > 11 else "N/A"
        })
    return results
```

## Expected Outputs

| Step | Output | Description |
|------|--------|-------------|
| Input Parse | Protein object | Loaded protein with sequence |
| UniProt Query | Protein + PDB refs | Sequence and cross-references |
| MSA | .a3m file | Multiple sequence alignment results |
| FoldSeek | .m8 file | Similar structures with scores |

## Error Handling

### Invalid Input Format

**Symptom**: Input not recognized as valid protein identifier or file.

**Solution**: Check input format and provide guidance.

```python
try:
    protein, has_structure, input_type = parse_input(user_input)
except Exception as e:
    print(f"Could not parse input: {e}")
    print("Supported formats: UniProt ID, PDB ID, FASTA file, PDB file, FASTA string")
```

### UniProt/PDB API Unavailable

**Symptom**: HTTP request fails with timeout or error status.

**Solution**: Use alternative database or cached data.

```python
try:
    protein, has_structure, input_type = query_uniprot(uniprot_id)
except requests.RequestException:
    print("UniProt API unavailable. Try providing FASTA sequence directly.")
```

### FoldSeek/MSA Server Busy

**Symptom**: Long wait time or rate limit error.

**Solution**: Wait and retry, or suggest alternative.

```python
# These tools have built-in retry logic with rate limiting
# If failing persistently, suggest using local BLAST or visiting web interface
```

## Interpretation Guide

### Sequence Identity (MSA)

| Identity | Relationship | Meaning |
|----------|--------------|---------|
| > 90% | Identical/Near-identical | Same protein, possibly different species |
| 70-90% | Homolog | Same family, likely similar function |
| 30-70% | Distant homolog | May share fold, function may differ |
| < 30% | Twilight zone | Relationship uncertain |

### FoldSeek E-value and Identity

| E-value | Identity | Significance |
|---------|----------|--------------|
| < 1e-50 | > 90% | Very high confidence, same fold |
| 1e-50 to 1e-10 | 50-90% | High confidence, similar structure |
| 1e-10 to 1e-3 | 30-50% | Moderate confidence, possible homolog |
| > 1e-3 | < 30% | Low confidence, may be random match |

## Example

```
Input: P0DTC2 (SARS-CoV-2 Spike protein)

Step 1: Detected UniProt ID
Step 2: Queried UniProt → Sequence (1273 aa) + 1961 PDB references
Step 3: Structure available. User chooses FoldSeek.
Step 4: Running FoldSeek on 6LZG (RBD domain)...

Output (Top 5 similar structures):
  Target                                           | Identity | E-value
  --------------------------------------------------|----------|---------
  SARS-CoV-2 Gamma RBD with ACE2 mutant             | 99.6%    | 1.6e-83
  SARS-CoV-2 Beta RBD with ACE2                     | 99.3%    | 6.9e-83
  SARS-CoV-2 Omicron RBD with ACE2                  | 99.3%    | 9.8e-81
  SARS coronavirus spike RBD (2003)                 | 100.0%   | 1.4e-80
  American mink ACE2 with RBD                       | 84.2%    | 6.3e-77
```

## See Also

- `examples/basic_example.py` - Full runnable example
- `references/databases.md` - Database descriptions (PDB, UniProt, AFDB)
