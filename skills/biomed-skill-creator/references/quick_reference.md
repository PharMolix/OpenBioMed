# BioMed Skill Creator Quick Reference

## Workflow Summary

```
1. Capture Intent → 2. Design Workflow → 3. Validate → 4. Finalize → 5. Evaluate
        ↑                                                      ↓
        └──────────────── Iterate if needed ←──────────────────┘
```

## Common BioMed Workflow Patterns

### Drug-Likeness Analysis
```
molecule_name_request
       ↓
┌──────┴──────┐
│  molecule_qed      │
│  molecule_sa       │  → Summarize
│  molecule_logp     │
│  molecule_lipinski │
└──────────────┘
```

### Protein Mutation Analysis
```
protein_uniprot_request
       ↓
mutation_explanation
       ↓
   Summarize
```

### Structure-Based Drug Design
```
protein_pdb_request
       ↓
extract_molecules_from_pdb_file
       ↓
structure_based_drug_design
       ↓
   Docking + Evaluation
```

### Molecule Q&A
```
molecule_name_request
       ↓
molecule_question_answering
       ↓
   Format Answer
```

## Input Type Reference

| Input | Factory Method | Example |
|-------|---------------|---------|
| Molecule by name | `TOOLS["molecule_name_request"]` | `name="aspirin"` |
| Molecule by SMILES | `Molecule.from_smiles()` | `"CC(=O)O"` |
| Protein by UniProt | `TOOLS["protein_uniprot_request"]` | `accession="P00533"` |
| Protein by PDB | `TOOLS["protein_pdb_request"]` | `accession="1ABC"` |
| Text | `Text.from_str()` | `"What is this?"` |

## Score Interpretation

| Metric | Range | Good | Excellent |
|--------|-------|------|-----------|
| QED | 0-1 | 0.5-0.7 | > 0.7 |
| SA | 1-10 | 3-6 | < 3 |
| LogP | - | -0.4 to 5.6 | 1-3 |
| Lipinski | 0-4 violations | 0-1 | 0 |
| Docking | kcal/mol | < -7 | < -9 |

## Evaluation Checklist

- [ ] 2-3 realistic test prompts
- [ ] Objective assertions
- [ ] Run with-skill and baseline
- [ ] Grade outputs
- [ ] Analyze patterns
- [ ] Iterate if needed

## Key Questions for Validation

1. "What input can I use to test this?"
2. "Does the output match expectations?"
3. "Should any steps be added/removed?"
4. "Are error cases handled?"
