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

Retrieve proteins with similar structures, sequences, or from the same family using MSA (sequence) or FoldSeek (structure) via the OpenBioMed API.

## When to Use

- User provides a protein and wants to find similar proteins
- User asks for homologs or orthologs of a protein
- User wants proteins with similar 3D structure
- User wants to search by sequence similarity
- User provides UniProt ID, PDB ID, FASTA sequence, or PDB file as input

## API Endpoint Resolution

The skill resolves the OpenBioMed API base URL in this order:

1. **Environment variable**: `${OPENBIOMED_API_BASE_URL}` (if set)
2. **Docker container default**: `http://openbiomed-server:8090` (if running in Docker)
3. **Local development default**: `http://127.0.0.1:8090`

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL.

## Workflow

### Step 1: Parse Input and Load Protein

Detect input type and prepare protein input for API call.

| Input Type | Example | How to Handle |
|------------|---------|---------------|
| PDB file path | `./protein.pdb` | Use path directly (must exist on server) |
| FASTA sequence | `MKFLILLFNILCLFPVLAADNH...` | Use sequence string directly |
| UniProt ID | `P0DTC2` | Query UniProt API to get sequence |
| PDB ID | `6LZG` | Query PDB API to get structure file |

**If input is UniProt ID**, first retrieve the sequence:

```bash
curl -s "https://rest.uniprot.org/uniprotkb/<UniProt_accession>?format=json" | jq -r '.sequence.value'
```

**If input is PDB ID**, first download the structure:

```bash
# Option 1: Use OpenBioMed protein_pdb_request (recommended)
curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "protein_pdb_request", "query": "<PDB_ID>", "mode": "file_only"}'

# Option 2: Direct download from RCSB PDB
curl -L -o protein.pdb "https://files.rcsb.org/download/<PDB_ID>.pdb"
```

### Step 2: Choose Search Method

| Search Type | Description | When to Use |
|-------------|-------------|-------------|
| `msa` | Sequence similarity (MMSeqs2/ColabFold) | Sequence-only input, finding homologs |
| `foldseek` | Structure similarity (FoldSeek) | Has 3D structure, finding similar folds |

If the input has 3D structure (PDB file or structure from PDB ID), ask user which method to use. Default to `foldseek` for structural inputs.

### Step 3: Call similar_protein_search API

#### MSA Search (Sequence Similarity)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "similar_protein_search", "search_type": "msa", "protein": "<FASTA_SEQUENCE>"}'
```

**Response**:
```json
{
  "task": "similar_protein_search",
  "search_type": "msa",
  "result_path": "./tmp/msa_results_xxx/uniref.a3m",
  "description": "MSA results saved to .a3m file"
}
```

#### FoldSeek Search (Structure Similarity)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "similar_protein_search", "search_type": "foldseek", "protein": "<PDB_FILE_PATH>"}'
```

**Optional**: Specify databases to search:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "similar_protein_search", "search_type": "foldseek", "protein": "<PDB_FILE_PATH>", "database": ["pdb100", "afdb50"]}'
```

Available databases: `pdb100`, `afdb50`, `afdb-swissprot`, `afdb-proteome`, `cath50`, `mgnify_esm30`

**Response**:
```json
{
  "task": "similar_protein_search",
  "search_type": "foldseek",
  "result_path": "./tmp/foldseek_results_xxx/result.m8",
  "result_dir": "./tmp/foldseek_results_xxx",
  "database": ["pdb100", "afdb50"],
  "description": "FoldSeek results saved to .m8 file"
}
```

### Step 4: Parse Results

#### MSA Results (.a3m file)

The `.a3m` file contains multiple sequence alignment. Parse to extract top hits:

```bash
# Read the first N hits from a3m file
head -n 20 "${result_path}"
# Each hit format: >hit_id\naligned_sequence
```

#### FoldSeek Results (.m8 file)

The `.m8` file is tab-separated with columns:

| Column | Description |
|--------|-------------|
| 0 | Query ID |
| 1 | Target ID |
| 2 | Sequence identity |
| 3 | Alignment length |
| 4-9 | Alignment details |
| 10 | Probability |
| 11 | E-value |

Parse to display top results:

```bash
# Sort by e-value (column 11), show top 10
cat "${result_path}" | sort -t'\t' -k11,11g | head -10

# Or use Python for structured output
python3 -c "
import pandas as pd
df = pd.read_csv('${result_path}', sep='\t', header=None)
print(df[[1, 2, 3, 11]].head(10).to_string(index=False, header=['Target', 'Identity', 'AlnLen', 'E-value']))
"
```

## Example Usage

### Example 1: Sequence Similarity Search

```
Input: "Find similar proteins to this sequence: MKFLILLFNILCLFPVLAADNH..."

Step 1: Detected FASTA sequence input
Step 2: No structure available → use MSA
Step 3: Call API

curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "similar_protein_search", "search_type": "msa", "protein": "MKFLILLFNILCLFPVLAADNH..."}'

Step 4: Parse .a3m results

Output (Top similar sequences):
  Target            | Identity | Description
  -------------------|----------|------------
  sp|P00533|EGFR_HUMAN | 95%    | Epidermal growth factor receptor
  sp|P04626|ERBB2_HUMAN | 78%   | Receptor tyrosine-protein kinase erbB-2
```

### Example 2: Structure Similarity Search

```
Input: "Find proteins with similar structure to PDB 6LZG"

Step 1: Download PDB file
  curl -X POST "${OPENBIOMED_API_BASE_URL}/web_search/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "protein_pdb_request", "query": "6LZG", "mode": "file_only"}'
  → protein file: ./tmp/protein_6lzg.pdb

Step 2: Structure available → use FoldSeek
Step 3: Call API

  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "similar_protein_search", "search_type": "foldseek", "protein": "./tmp/protein_6lzg.pdb"}'

Step 4: Parse .m8 results

Output (Top similar structures):
  Target                      | Identity | E-value
  ----------------------------|----------|---------
  6LZG_A (SARS-CoV-2 RBD)     | 100.0%   | 0.0
  6M0J_B (SARS-CoV RBD)       | 89.2%    | 1.4e-80
  7A2N_A (Mink ACE2+RBD)      | 84.2%    | 6.3e-77
```

## Expected Outputs

| Step | Output | Description |
|------|--------|-------------|
| Input Parse | Protein ready | FASTA string or PDB file path |
| API Call | result_path | Path to results file |
| MSA | .a3m file | Multiple sequence alignment with homologs |
| FoldSeek | .m8 file | Similar structures with identity/e-value |

## Error Handling

### Invalid Input Format

**Symptom**: API returns error about invalid protein format.

**Solution**: Check input format. Supported formats:
- FASTA sequence string (letters only)
- PDB file path (must exist on server filesystem)
- Use `protein_uniprot_request` or `protein_pdb_request` to convert IDs first

### API Unavailable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable:
```bash
curl "${OPENBIOMED_API_BASE_URL}/healthz"
# Should return "Service available"
```

### Search Timeout

**Symptom**: Long wait time or timeout error from MSA/FoldSeek.

**Solution**: These external services may be busy. Wait and retry, or use smaller sequence/structure input.

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

## See Also

- `protein_uniprot_request` - Get protein sequence from UniProt ID
- `protein_pdb_request` - Get protein structure from PDB ID
- `protein_folding` - Predict protein structure from sequence (for MSA → FoldSeek workflow)