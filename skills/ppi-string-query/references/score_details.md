# STRING Score Methodology

## Overview

STRING integrates multiple evidence channels to compute protein-protein interaction confidence scores. Each channel contributes differently based on reliability and coverage.

## Score Types

### Combined Score (`score`)

The combined score represents the overall confidence that two proteins interact. It integrates all evidence channels with weighted contributions:

```
combined_score = 1 - Π(1 - channel_score)
```

This probabilistic integration means that independent evidence from multiple channels increases confidence more than repeated evidence from a single channel.

### Experimental Evidence (`escore`)

**Source**: High-throughput and curated experiments
- BioGRID, IntAct, MINT, DIP databases
- Yeast two-hybrid, co-immunoprecipitation, affinity purification
- In vitro binding assays

**Interpretation**:
- > 0.7: Strong experimental support
- 0.3-0.7: Moderate experimental evidence
- < 0.3: Weak or indirect evidence

### Text Mining (`tscore`)

**Source**: Literature co-occurrence
- PubMed abstract mining
- Full-text article analysis
- GeneRIF annotations

**Interpretation**:
- High scores indicate frequent co-mentioning in scientific literature
- May reflect functional relationships rather than direct binding
- Useful for discovering less-studied interactions

### Database Evidence (`dscore`)

**Source**: Curated pathway databases
- KEGG pathways
- Reactome
- BioCyc
- MetaCyc

**Interpretation**:
- High confidence, manually curated
- Often reflects pathway membership rather than direct binding
- Good for identifying functional modules

### Co-expression (`ascore`)

**Source**: Gene expression correlation
- GEO, ArrayExpress datasets
- RNA-seq expression profiles
- Co-regulation patterns

**Interpretation**:
- High correlation suggests functional relationship
- Does not guarantee physical interaction
- Tissue-specific patterns may be relevant

### Phylogenetic Profiles (`pscore`)

**Source**: Evolutionary conservation
- Co-occurrence across genomes
- Gene neighborhood conservation
- Domain fusion events

**Interpretation**:
- Suggests functional coupling
- Works best for metabolic enzymes
- Less reliable for signaling proteins

### Gene Fusion (`fscore`)

**Source**: Fusion events across genomes
- Rosetta Stone method
- Domain-level fusions

**Interpretation**:
- Strong evidence for functional interaction
- Rare but high-confidence when present
- More common in prokaryotes

### Genomic Neighborhood (`nscore`)

**Source**: Gene proximity
- Operon membership (prokaryotes)
- Syntenic regions (eukaryotes)

**Interpretation**:
- Very useful for bacterial proteins
- Less informative for human proteins
- Suggests co-regulation

## Confidence Thresholds

STRING defines four confidence levels based on combined scores:

| Threshold | Score | Description |
|-----------|-------|-------------|
| Low | 0.150 | Most inclusive, many false positives |
| Medium | 0.400 | Balanced sensitivity/specificity |
| High | 0.700 | Reliable interactions, recommended default |
| Highest | 0.900 | Very confident, may miss true positives |

## Best Practices

1. **Start with high confidence (700)**: Default provides reliable results with manageable false positives.

2. **Check multiple channels**: Interactions with high scores in multiple channels are more reliable than those supported by single evidence type.

3. **Consider your context**:
   - Structural studies: prioritize `escore` and `dscore`
   - Systems biology: all channels relevant
   - Drug target discovery: `escore` and `dscore` most valuable

4. **Cross-reference with other databases**: STRING aggregates data; verify critical interactions in primary sources.

5. **Batch queries**: For network analysis, query multiple proteins and build the network locally.

## Limitations

- **Indirect interactions**: STRING includes both physical and functional associations
- **Species coverage**: Human and model organisms best covered; exotic species may have sparse data
- **Bias toward well-studied proteins**: Popular proteins have more evidence by literature
- **False negatives**: Novel interactions may lack evidence across all channels

## References

- Szklarczyk D, et al. (2023) The STRING database in 2023. Nucleic Acids Res. 51(D1):D638-D646
- STRING Documentation: https://string-db.org/help/
