# Regeneration Strategies

This document describes strategies for handling cases where insufficient candidates meet user criteria.

## When Regeneration Is Needed

Regeneration is triggered when:
1. No candidates pass all filtering criteria
2. Fewer than target number of diverse leads are selected
3. All selected leads fail a specific criterion (e.g., all have high side effects)

## Failure Analysis

### Step 1: Identify Primary Failure Mode

```python
fail_stats = {'docking': 0, 'qed': 0, 'lipinski': 0, 'side_effects': 0}

for mol in candidates:
    if mol.docking_score > docking_threshold:
        fail_stats['docking'] += 1
    if mol.qed < qed_min:
        fail_stats['qed'] += 1
    if mol.lipinski < lipinski_min:
        fail_stats['lipinski'] += 1
    if mol.num_side_effects > side_effects_max:
        fail_stats['side_effects'] += 1

# Identify the biggest bottleneck
primary_failure = max(fail_stats, key=fail_stats.get)
```

### Step 2: Determine Appropriate Response

| Primary Failure | Likely Cause | Recommended Action |
|-----------------|--------------|-------------------|
| Lipinski | Target class (kinases) | Relax Lipinski criterion |
| Side Effects | Large, complex molecules | Relax side effects threshold |
| QED | Non-drug-like scaffolds | Generate more candidates |
| Docking | Poor pocket fit | Check pocket definition |

## Strategy 1: Criteria Relaxation

### Conservative Relaxation

Small adjustments to criteria:

| Criterion | Original | Relaxed | Impact |
|-----------|----------|---------|--------|
| Docking | -10 | -8 | +10-20% candidates |
| QED | 0.5 | 0.4 | +10-15% candidates |
| Lipinski | 4 | 3 | +20-40% candidates |
| Side Effects | 15 | 18 | +15-25% candidates |

### Progressive Relaxation

Iteratively relax the most restrictive criterion:

```python
def progressive_relaxation(candidates, criteria, target):
    relaxation_steps = [
        ('side_effects_max', 3),   # Add 3 to max
        ('lipinski_min', -1),       # Allow 1 more violation
        ('qed_min', -0.1),          # Lower QED by 0.1
        ('docking_threshold', 2),   # Allow +2 kcal/mol
    ]

    for criterion, adjustment in relaxation_steps:
        criteria[criterion] += adjustment
        selected = apply_criteria(candidates, criteria)
        if len(selected) >= target:
            return selected, criteria

    return selected, criteria
```

### Target-Specific Relaxation

Adjust based on known target class properties:

| Target Class | Typical Lipinski | Typical QED | Notes |
|--------------|------------------|-------------|-------|
| Kinase inhibitors | 1-2 rules | 0.3-0.5 | Large, flat molecules |
| GPCR ligands | 2-3 rules | 0.4-0.6 | Moderate size |
| Antibiotics | 1-2 rules | 0.3-0.5 | Often violate rules |
| Protease inhibitors | 2-3 rules | 0.4-0.6 | Peptidomimetic |

## Strategy 2: Generate More Candidates

### When to Generate More

- All criteria have reasonable thresholds
- Failure is distributed across criteria (not one bottleneck)
- More candidates may find better solutions

### How Many to Generate

```python
deficit = target_leads - current_leads
additional_to_generate = deficit * 3  # Rule of thumb: 3x deficit

# Account for filtering rate
pass_rate = current_passed / total_generated
if pass_rate > 0:
    additional_to_generate = int(deficit / pass_rate)
```

### Different Generation Strategies

1. **Different seeds**: Change random seed for MolCraft
2. **Longer sampling**: Increase `num_sample_steps` from 100 to 200
3. **Different pocket radius**: Expand pocket slightly
4. **Fragment-based**: Start from known fragments

## Strategy 3: Multi-Iteration Workflow

### Iterative Improvement Loop

```
Iteration 1: Generate 40 → Filter → 5 leads
     ↓
Iteration 2: Generate 40 more → Filter from 85 total → 12 leads
     ↓
Iteration 3: Relax criteria → Filter → 18 leads
     ↓
Accept or continue
```

### Implementation

```python
def iterative_lead_design(target, criteria, target_leads, max_iterations=5):
    all_candidates = []

    for iteration in range(max_iterations):
        # Generate new candidates
        new_candidates = generate_candidates(target, num=40)
        all_candidates.extend(new_candidates)

        # Apply filtering
        selected = filter_and_select(all_candidates, criteria)

        if len(selected) >= target_leads:
            return selected[:target_leads]

        # Offer relaxation if not first iteration
        if iteration > 0:
            print(f"Current: {len(selected)}, Target: {target_leads}")
            choice = user_input("Options: 1) More, 2) Relax, 3) Accept")
            if choice == "relax":
                criteria = relax_criteria(criteria)

    return selected
```

## Strategy 4: Alternative Approaches

### When Generation Fails Completely

If MolCraft produces no candidates meeting any reasonable criteria:

1. **Check pocket definition**: Pocket may be too small or incorrect
2. **Try different PDB structure**: Different conformation may help
3. **Use reference ligand modification**: Edit known inhibitor instead
4. **Switch to virtual screening**: Screen existing compound libraries

### Fallback Workflow

```python
if len(filtered_candidates) == 0:
    # Option A: Modify reference ligand
    edited_mols = text_based_molecule_editing(
        reference_ligand,
        text="Improve drug-likeness while maintaining binding")

    # Option B: Virtual screening
    similar_mols = search_similar_molecules(reference_ligand, threshold=0.7)

    # Option C: Scaffold hopping
    new_scaffolds = scaffold_hop(reference_ligand)
```

## Decision Matrix

| Scenario | Best Strategy |
|----------|---------------|
| One criterion dominates failures | Relax that criterion |
| Failures distributed evenly | Generate more candidates |
| No candidates after 100+ generated | Check pocket/target suitability |
| Time-constrained | Accept partial results, prioritize |
| Need exact target count | Progressive relaxation |

## Communication with User

### Clear Status Reporting

```
REGENERATION STATUS
────────────────────
Current leads: 6 of 20 target (30%)

Failure Analysis:
  • Side effects > 18: 35 candidates (87%)
  • Lipinski < 4 rules: 10 candidates (25%)
  • QED < 0.4: 8 candidates (20%)
  • Docking > -10: 2 candidates (5%)

Primary bottleneck: Side Effects

RECOMMENDED ACTIONS:
1. Relax side_effects_max to 22 (likely +15 candidates)
2. Generate 40 more candidates
3. Accept current 6 leads for initial screening

Your choice? [1/2/3]: _
```

### Setting Expectations

- First iteration rarely achieves target
- 2-3 iterations are typical
- Some targets are inherently difficult
- Final count may be lower than ideal
