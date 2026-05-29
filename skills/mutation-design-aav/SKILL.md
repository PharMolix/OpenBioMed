---
name: mutation-design-aav
description: >
  Design high-fitness AAV VP1 capsid protein mutants through multi-round iterative optimization.
  Use this skill when:
  (1) Designing AAV mutants with improved DNA packaging fitness,
  (2) Running computational iterative directed evolution,
  (3) Performing fast mutation search guided by an oracle model.
license: MIT
category: protein-engineering
tags: [mutation-design, aav, directed-evolution, protein-optimization]
---

# AAV Mutation Design

Design high-fitness AAV VP1 capsid protein mutants through multi-round iterative optimization using the OpenBioMed API.

## When to Use

- User wants to design AAV mutants with improved DNA packaging fitness
- User asks for computational directed evolution optimization
- User wants to generate AAV variants with high fitness and diversity
- User requests multi-round mutation search

## API Endpoint Resolution

The skill resolves the OpenBioMed API base URL in this order:

1. **Environment variable**: `${OPENBIOMED_API_BASE_URL}` (if set)
2. **Docker container default**: `http://openbiomed-server:8090` (if running in Docker)
3. **Local development default**: `http://127.0.0.1:8090`

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL.

## Workflow

### Step 1: Call mutation_design_aav API

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "mutation_design_aav"}'
```

**Optional Parameters**:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"task": "mutation_design_aav", "num_rounds": 10, "population_size": 96, "max_mutations": 4, "diversity_weight": 0.1}'
'
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
  "task": "mutation_design_aav",
  "csv_file": "./tmp/mutation_design_aav/aav_mutants_xxx.csv",
  "description": "AAV mutation design completed. Generated 96 mutants with best fitness 0.85. Results saved to ./tmp/mutation_design_aav/aav_mutants_xxx.csv"
}
```

### Step 2: Parse Results

The output CSV file contains two columns:

| Column | Description |
|--------|-------------|
| sequence | AAV mutant sequence (28 amino acids) |
| fitness | Predicted DNA packaging fitness score |

Example CSV content:

```
sequence,fitness
ADMEIIQVNPYSSEQYGDVATPLYHGTG,0.96
ADMEIRQVNPYSSEQYGDVATPLQHGTG,0.93
ADSELASTNPVSTELYGIVATNLMAQAS,0.92
...
```

The CSV contains exactly 96 sequences sorted by fitness in descending order.

## Example Usage

### Example 1: Basic AAV Mutation Design

```
Input: "Design AAV mutants with higher DNA packaging fitness"

Step 1: Call API with default parameters
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "mutation_design_aav"}'

Output:
  CSV file: ./tmp/mutation_design_aav/aav_mutants_123456.csv
  96 mutants with fitness scores
  Best fitness: 0.85
```

### Example 2: Custom Parameters

```
Input: "Generate 50 AAV mutants with up to 3 mutations each"

Step 1: Call API with custom parameters
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "mutation_design_aav", "population_size": 50, "max_mutations": 3}'

Output:
  CSV file with 50 mutants
  Optimization focused on fewer mutations
```

### Example 3: Higher Diversity

```
Input: "Design diverse AAV mutants"

Step 1: Call API with higher diversity weight
  curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"task": "mutation_design_aav", "diversity_weight": 0.3}'

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

1. **Initial Sequences**: Download from pre-defined URL (28-amino acid VP1 segment)
2. **Oracle Model**: CNN model for fitness prediction
3. **ESM2 Embeddings**: Sequence representation for optimization
4. **Mutation Strategy**: Point mutations only (≤4 per sequence)
5. **Diversity Metric**: Average pairwise Hamming distance
6. **Stopping Criteria**: 10 rounds or 3 rounds without improvement

### Fitness Score Interpretation

| Fitness Range | Interpretation |
|---------------|----------------|
| 0.8 - 1.0 | High fitness, good DNA packaging potential |
| 0.6 - 0.8 | Moderate fitness, reasonable improvement |
| 0.4 - 0.6 | Low fitness, may need further optimization |
| < 0.4 | Poor fitness, unlikely to be functional |

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
Should I use mutation_design_aav?
│
└─ What protein are you designing?
   ├─ AAV VP1 capsid protein → mutation-design-aav ✓
   ├─ GFP → mutation-design-gfp
   └─ General protein → functional-protein-design
```

## Next Steps

After AAV mutation design:
- **Sequence Analysis**: Analyze mutation patterns and positions
- **Validation**: Experimentally validate top candidates
- **Combination**: Combine beneficial mutations from different candidates

## See Also

- `mutation-design-gfp` - Design GFP mutants with higher fluorescence
- `functional-protein-design` - General functional protein design
- `protein-mutation-analysis` - Analyze protein mutations