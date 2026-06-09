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

Retrieve proteins with similar structures, sequences, or from the same family using MSA (sequence) or FoldSeek (structure).

## When to Use

- User provides a protein and wants to find similar proteins
- User asks for homologs or orthologs of a protein
- User wants proteins with similar 3D structure
- User wants to search by sequence similarity
- User provides UniProt ID, PDB ID, FASTA sequence, or PDB file as input

## API Endpoints

### MSA Service (Sequence Similarity)

Remote MSA service for fast asynchronous sequence search.

**Base URL**: `${MSA_API_BASE_URL}` (environment variable)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/msa/search/submit` | POST | Submit MSA job (returns job_id immediately) |
| `/msa/search/status/{job_id}` | GET | Poll job status |
| `/msa/search/result/{job_id}` | GET | Fetch results when completed |

### OpenBioMed API (Structure Similarity)

**Base URL**: `${OPENBIOMED_API_BASE_URL}` (resolved in order: env var → Docker default `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520` → local `http://127.0.0.1:8090`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_pipeline/` | POST | Run FoldSeek structure search |

## Workflow

### Step 1: Parse Input and Load Protein

Detect input type and prepare protein input.

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

| Search Type | Description | When to Use | Expected Runtime |
|-------------|-------------|-------------|------------------|
| `msa` | Sequence similarity (self-hosted service) | Sequence-only input, finding homologs | **2–5+ 分钟**（取决于序列长度和服务器负载）|
| `foldseek` | Structure similarity (FoldSeek) | Has 3D structure, finding similar folds | **10–30 秒**（单数据库），多数据库搜索可能更长 |

⚠️ **MSA 搜索耗时较长**：MSA 模式需要远程调用服务，典型耗时 2–5 分钟，长序列或服务器繁忙时可能超过 10 分钟。调用 API 后请耐心等待，不要重复提交。FoldSeek 通常较快，但如果搜索多个大型数据库（如 `afdb-swissprot`）也可能需要 1–2 分钟。

If the input has 3D structure (PDB file or structure from PDB ID), ask user which method to use. Default to `foldseek` for structural inputs.

### Step 3: Execute Search

#### MSA Search (Sequence Similarity) — Direct API Call

The MSA service uses a **submit-job + poll-status + fetch-result** pattern. You must execute these steps manually:

##### 3.1 Submit MSA Job

Submit the sequence to get a job_id (returns immediately, no timeout):

> ⏱️ **注意：MSA 搜索通常需要 2–5 分钟，请耐心等待响应。**

```bash
curl -s -X POST "${MSA_API_BASE_URL}/msa/search/submit" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "<FASTA_SEQUENCE>"}'
```

**Response**:
```json
{
  "job_id": "58d375eb-c738-4621-8554-8821c303934e",
  "status": "PENDING",
  "message": "Job submitted successfully."
}
```

Extract the `job_id` for subsequent polling.

##### 3.2 Poll Job Status

Poll until status becomes `COMPLETED` or `FAILED`:

```bash
curl -s "${MSA_API_BASE_URL}/msa/search/status/${JOB_ID}"
```

**Status Values**:

| Status | Meaning |
|--------|---------|
| `PENDING` | Job queued, waiting to start |
| `RUNNING` | Jackhmmer search in progress |
| `COMPLETED` | Search finished successfully |
| `FAILED` | Search failed |

**Polling Response Examples**:

```json
// Running
{
  "job_id": "58d375eb-...",
  "status": "RUNNING",
  "message": "Jackhmmer search in progress.",
  "elapsed_seconds": null
}

// Completed
{
  "job_id": "58d375eb-...",
  "status": "COMPLETED",
  "message": "Search finished successfully.",
  "elapsed_seconds": 651.77
}

// Failed
{
  "job_id": "58d375eb-...",
  "status": "FAILED",
  "message": "Search failed."
}
```

**Polling Strategy**:
- Poll every 60 seconds
- Continue until `status == "COMPLETED"` and `message == "Search finished successfully."`
- If `status == "FAILED"`, report the error

##### 3.3 Fetch MSA Results

When job is completed, fetch the results directly from API:

```bash
curl -s "${MSA_API_BASE_URL}/msa/search/result/${JOB_ID}"
```

**Result JSON Structure**:
```json
{
  "job_id": "58d375eb-...",
  "status": "COMPLETED",
  "result": {
    "unpaired_msa": ">Original query\nMKTAYIAK...\n>hit_1\n...\n",
    "paired_msa": ">Original query\nMKTAYIAK...\n>hit_1\n...\n",
    "databases": {
      "uniref90": {
        "a3m": ">Original query\nMKTAYIAK...\n...",
        "depth": 2818
      },
      "mgnify": {
        "a3m": ">Original query\nMKTAYIAK...\n...",
        "depth": 5000
      },
      "small_bfd": {
        "a3m": ">Original query\nMKTAYIAK...\n...",
        "depth": 51
      },
      "uniprot_cluster_annot": {
        "a3m": ">Original query\nMKTAYIAK...\n...",
        "depth": 7729
      }
    },
    "elapsed_seconds": 651.77
  }
}
```

**Key Fields**:
- `result.unpaired_msa` — Primary MSA alignment result (A3M format string)
- `result.paired_msa` — Paired alignment with curated sequences
- `result.elapsed_seconds` — Total search time

**Note**: The `depth` field in `databases` may show 0 even when hits exist. Use `unpaired_msa` or `paired_msa` content to determine actual hit count.

#### FoldSeek Search (Structure Similarity)

Call the OpenBioMed API for structure search:

> ⏱️ FoldSeek 通常 10–30 秒返回结果，搜索多个大型数据库时可能需要 1–2 分钟。

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
  "description": "FoldSeek results saved to .m8 file"
}
```

Parse the `.m8` file:

```bash
# Tab-separated: Query ID, Target ID, Identity, AlnLen, ..., E-value
cat "${result_path}" | sort -t'\t' -k11,11g | head -10
```

## Example Usage

### Example 1: MSA Sequence Search (Manual API Flow)

```
Input: "Find similar proteins to sequence MKTAYIAKQRQISFVK..."

Step 1: Detected FASTA sequence input
Step 2: No structure available → use MSA
Step 3: Execute MSA search manually

  # 3.1 Submit job
  curl -s -X POST "${MSA_API_BASE_URL}/msa/search/submit" \
    -H "Content-Type: application/json" \
    -d '{"sequence": "MKTAYIAKQRQISFVK..."}'
  → job_id: "58d375eb-c738-4621-8554-8821c303934e"

  # 3.2 Poll status (repeat until COMPLETED)
  # Note: MSA jobs typically take ~120 seconds, poll at 60s intervals
  curl -s "${MSA_API_BASE_URL}/msa/search/status/58d375eb-..."
  → status: "RUNNING" (wait 60s, poll again)
  → status: "COMPLETED", message: "Search finished successfully."

  # 3.3 Fetch and parse results directly from API
  curl -s "${MSA_API_BASE_URL}/msa/search/result/58d375eb-..."

  # Parse unpaired_msa to show top hits and total count
  → Top hit: UniRef90_Q8ZKW4 Aspartate--ammonia ligase (Salmonella typhi)
  → Total hits: ~7500 sequences
  → Elapsed time: 146 seconds

Output: MSA found ~7500 similar sequences, primarily Aspartate--ammonia ligase homologs from Enterobacteriaceae
```

### Example 2: FoldSeek Structure Search

```
Input: "Find proteins with similar structure to PDB 6LZG"

Step 1: Download PDB file
  curl -L -o protein.pdb "https://files.rcsb.org/download/6LZG.pdb"

Step 2: Structure available → use FoldSeek
Step 3: Call OpenBioMed API

  curl -X POST "http://127.0.0.1:8090/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "similar_protein_search", "search_type": "foldseek", "protein": "./protein.pdb"}'

Step 4: Parse .m8 results

Output (Top similar structures):
  Target                      | Identity | E-value
  ----------------------------|----------|---------
  6LZG_A (SARS-CoV-2 RBD)     | 100.0%   | 0.0
  6M0J_B (SARS-CoV RBD)       | 89.2%    | 1.4e-80
```

## Expected Outputs

| Method | Output | Format |
|--------|--------|--------|
| MSA | JSON with `unpaired_msa`/`paired_msa` A3M strings | API response |
| FoldSeek | JSON with `result_path` to `.m8` file | API response |

**For Remote Agents**: Results are returned directly in API response. Parse JSON to extract alignment data. File paths in FoldSeek results are server-side paths that require additional API calls to retrieve content.

## Error Handling

### MSA Job Failed

**Symptom**: Status returns `FAILED`.

**Solution**: Check the `message` field for error details. Common causes:
- Invalid sequence format
- MSA service temporarily unavailable

### MSA Result Not Ready

**Symptom**: `/msa/search/result/{job_id}` returns 404.

**Solution**: Job may still be running. Verify status is `COMPLETED` before fetching results.

### FoldSeek API Unavailable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint:
```bash
curl "${OPENBIOMED_API_BASE_URL}/healthz"
# Should return "Service available"
```

### Search Timeout

**Symptom**: Long wait time or timeout error from MSA/FoldSeek.

**Solution**: These external services may be busy. MSA search typically takes 2–5 minutes, FoldSeek 10–30 seconds. If waiting exceeds 10 minutes for MSA or 2 minutes for FoldSeek, the remote service may be unavailable — wait and retry later, or use smaller sequence/structure input.
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