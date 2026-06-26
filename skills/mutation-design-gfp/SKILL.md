---
name: mutation-design-gfp
description: >
  Design high-fluorescence GFP mutants through multi-round iterative optimization.
  Use this skill when:
  (1) Designing GFP mutants with improved fluorescence,
  (2) Running computational iterative directed evolution,
  (3) Performing fast mutation search guided by an oracle model.
license: MIT
category: protein-engineering
tags: [mutation-design, gfp, directed-evolution, protein-optimization]
---

# GFP Mutation Design

Design high-fluorescence Green Fluorescent Protein (GFP) mutants through multi-round iterative optimization using the OpenBioMed API.

## When to Use

- User wants to design GFP mutants with improved fluorescence
- User asks for computational directed evolution optimization
- User wants to generate GFP variants with high fluorescence and diversity
- User requests multi-round mutation search

## API Endpoint Resolution

The skill resolves the OpenBioMed API base URL in this order:

1. **Environment variable**: `${OPENBIOMED_API_BASE_URL}` (if set)
2. **Docker container default**: `http://openbiomed-server:8090` (if running in Docker)
3. **Local development default**: `http://127.0.0.1:8090`

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL.

## Workflow

### Step 1: Call mutation_design_gfp API

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "mutation_design_gfp"}'
```

**Optional Parameters**:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "mutation_design_gfp", "num_rounds": 10, "population_size": 96, "max_mutations": 4, "diversity_weight": 0.1}'
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_rounds | int | 10 | Number of optimization rounds |
| population_size | int | 96 | Number of mutants per round |
| max_mutations | int | 4 | Max point mutations per sequence |
| diversity_weight | float | 0.1 | Weight for diversity in selection |

**Response**:

```json
{
  "task": "mutation_design_gfp",
  "csv_file": "./tmp/mutation_design_gfp/gfp_mutants_xxx.csv",
  "description": "GFP mutation design completed. Generated 96 mutants with best fitness 3.12. Results saved to ./tmp/mutation_design_gfp/gfp_mutants_xxx.csv"
}
```

### Step 2: Retrieve CSV Content

The `csv_file` returned is a server-side path. Use `read_csv_file` to get the content:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "read_csv_file", "value": "./tmp/mutation_design_gfp/gfp_mutants_xxx.csv"}'
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| value | string | required | CSV file path (from mutation_design_gfp response) |
| num_rounds | int | 100 | Max rows to return (reused as max_rows) |

**Response**:

```json
{
  "task": "read_csv_file",
  "csv_content": "sequence,fitness\nSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDAT...,3.12\n...",
  "data": [{"sequence": "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDAT...", "fitness": 3.12}, ...],
  "num_rows": 96,
  "total_rows": 96,
  "description": "CSV content read from ./tmp/mutation_design_gfp/gfp_mutants_xxx.csv: 96 rows returned"
}
```

### Step 3: Parse Results

The output CSV file contains two columns:

| Column | Description |
|--------|-------------|
| sequence | GFP mutant sequence (237 amino acids) |
| fitness | Predicted fluorescence fitness score (higher = brighter) |

Example CSV content:

```
sequence,fitness
SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDAT...,3.1207
SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDAT...,2.9841
...
```

The CSV contains exactly 96 sequences sorted by fitness in descending order. Fitness values are the raw oracle-model outputs (same scale as the training ground-truth fluorescence), so they are not bounded to [0, 1].

## Example Usage

### Example 1: Basic GFP Mutation Design

```
Input: "Design GFP mutants with higher fluorescence"

Step 1: Call API with default parameters
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "mutation_design_gfp"}'

Output:
  CSV file: ./tmp/mutation_design_gfp/gfp_mutants_123456.csv
  96 mutants with fitness scores
```

### Example 2: Custom Parameters

```
Input: "Generate 50 GFP mutants with up to 3 mutations each"

Step 1: Call API with custom parameters
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "mutation_design_gfp", "population_size": 50, "max_mutations": 3}'

Output:
  CSV file with 50 mutants
  Optimization focused on fewer mutations
```

### Example 3: Higher Diversity

```
Input: "Design diverse GFP mutants"

Step 1: Call API with higher diversity weight
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "mutation_design_gfp", "diversity_weight": 0.3}'

Output:
  CSV file with 96 diverse mutants
  Higher sequence diversity among top mutants
```

## Expected Outputs

| Output | Type | Description |
|--------|------|-------------|
| csv_file | string | Path to results CSV file |
| description | string | Human-readable summary |

## Technical Details

### Optimization Algorithm

1. **Initial Sequences**: Downloaded from a pre-defined URL (237-amino acid GFP variants)
2. **Oracle Model**: BaseCNN fitness predictor (GGS framework), trained with `monitor: val/spearmanr`; scores its own training data at Spearman ~0.87
3. **Mutation Strategy**: Point mutations only (≤4 per sequence)
4. **Diversity Metric**: Average pairwise Hamming distance
5. **Stopping Criteria**: 10 rounds or 3 rounds without improvement

### Fitness Score Interpretation

Fitness is the raw oracle output on the same scale as the training fluorescence ground truth (typically ~0.5–3.5). Higher means brighter predicted fluorescence. Because the bundled starting pool is a harder, out-of-distribution slice, absolute values on unseen mutants are approximate — trust the **ranking** (top rows are predicted brighter), not the exact magnitude.

## Error Handling

### API Unavailable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable:
```bash
curl "${OPENBIOMED_API_BASE_URL}/healthz"
# Should return "Service available"
```

### Oracle Model Download Failed

**Symptom**: API returns error about model download.

**Solution**: The tool will use a fallback scoring function. For accurate results, ensure the oracle model URLs are accessible.

### Empty Results

**Symptom**: CSV file contains fewer than 96 sequences.

**Solution**: Check the logs for optimization errors. The tool may have stopped early due to convergence.

## Decision Tree

```
Should I use mutation_design_gfp?
│
└─ What protein are you designing?
   ├─ GFP → mutation-design-gfp ✓
   ├─ AAV VP1 capsid protein → mutation-design-aav
   └─ General protein → functional-protein-design
```

## Next Steps

After GFP mutation design:
- **Sequence Analysis**: Analyze mutation patterns and positions
- **Validation**: Experimentally validate top candidates (measure actual fluorescence)
- **Combination**: Combine beneficial mutations from different candidates

## See Also

- `mutation-design-aav` - Design AAV mutants with higher DNA packaging fitness
- `functional-protein-design` - General functional protein design
- `protein-mutation-analysis` - Analyze protein mutations
