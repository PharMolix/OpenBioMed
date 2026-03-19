# Property Interpretation Guide

This document provides detailed interpretation guidelines for all properties calculated during lead design.

## Binding Affinity (Docking Score)

### What It Measures
AutoDock Vina predicts the binding free energy (ΔG) between a ligand and protein in kcal/mol.

### Score Interpretation

| Score (kcal/mol) | Binding Strength | Typical Use |
|------------------|------------------|-------------|
| < -12 | Very strong | Lead compounds, high potency |
| -12 to -10 | Strong | Good lead candidates |
| -10 to -7 | Moderate | Starting points for optimization |
| -7 to -5 | Weak | May need significant optimization |
| > -5 | Very weak | Unlikely to be useful |

### Important Notes
- Scores are predictions, not experimental values
- Different proteins have different typical score ranges
- Kinase inhibitors often show scores between -8 to -14 kcal/mol
- Consider scores relative to known inhibitors for the same target

---

## QED (Quantitative Estimate of Drug-likeness)

### What It Measures
QED combines 8 molecular properties into a single score (0-1) indicating drug-likeness:
- Molecular weight, LogP, HBD, HBA
- Polar surface area, rotatable bonds
- Aromatic rings, structural alerts

### Score Interpretation

| QED Score | Assessment | Recommendation |
|-----------|------------|----------------|
| > 0.7 | Excellent | Highly drug-like, prioritize |
| 0.5 - 0.7 | Good | Acceptable for most programs |
| 0.4 - 0.5 | Marginal | May need optimization |
| < 0.4 | Poor | Consider structural modification |

### Typical Values
- Aspirin: 0.55
- Ibuprofen: 0.54
- Caffeine: 0.46
- Many kinase inhibitors: 0.3-0.5 (often larger molecules)

---

## SA Score (Synthetic Accessibility)

### What It Measures
SA score estimates how difficult a molecule is to synthesize (1-10 scale).

### Score Interpretation

| SA Score | Difficulty | Typical Timeline |
|----------|------------|------------------|
| 1-3 | Easy | Days to weeks |
| 3-4 | Moderate | Weeks |
| 4-6 | Challenging | Weeks to months |
| 6-8 | Difficult | Months |
| > 8 | Very difficult | May not be synthesizable |

### Factors Affecting SA Score
- Complex ring systems increase score
- Many stereocenters increase score
- Unusual functional groups increase score
- Known scaffolds decrease score

---

## LogP (Lipophilicity)

### What It Measures
LogP measures the partition coefficient between octanol and water, indicating lipophilicity.

### Score Interpretation

| LogP Range | Behavior | Considerations |
|------------|----------|----------------|
| < -0.4 | Very hydrophilic | Poor membrane permeability |
| -0.4 to 5.6 | Optimal | Good oral bioavailability |
| 5.6 to 7.0 | Lipophilic | May have solubility issues |
| > 7.0 | Very lipophilic | Poor solubility, metabolism issues |

### Target-Specific Considerations
- CNS drugs: LogP 2-3 preferred
- Oral drugs: LogP 1-4 typical
- Kinase inhibitors: Often higher LogP (3-6) due to size

---

## Lipinski's Rule of Five

### What It Measures
Four rules for oral drug-likeness (rules obeyed, not violations):

| Rule | Threshold | Rationale |
|------|-----------|-----------|
| MW | ≤ 500 Da | Membrane permeability |
| LogP | ≤ 5 | Lipophilicity balance |
| HBD | ≤ 5 | Hydrogen bonding |
| HBA | ≤ 10 | Hydrogen bonding |

### Interpretation

| Rules Obeyed | Violations | Assessment |
|--------------|------------|------------|
| 4 | 0 | Perfect compliance |
| 3 | 1 | Acceptable, many drugs have 1 |
| 2 | 2 | Marginal, consider carefully |
| < 2 | > 2 | Likely issues with oral bioavailability |

### Exceptions
- Antibiotics often violate rules
- Kinase inhibitors frequently have 2-4 violations
- Natural products often violate rules
- Consider the specific target class

---

## BBB Penetration

### What It Measures
Probability that a compound crosses the blood-brain barrier.

### Interpretation

| Probability | Prediction | Use Case |
|-------------|------------|----------|
| > 0.7 | High likelihood | CNS-active drugs |
| 0.5 - 0.7 | Moderate likelihood | May reach brain |
| 0.3 - 0.5 | Low likelihood | Limited CNS exposure |
| < 0.3 | Unlikely | Peripheral-only action |

### Considerations
- CNS targets (Alzheimer's, Parkinson's): Want BBB penetration
- Peripheral targets: May want to avoid CNS side effects
- Many kinase inhibitors have low BBB penetration by design

---

## Side Effects (SIDER)

### What It Measures
Predictions for 27 side effect categories from the SIDER database.

### Categories Include
- Hepatobiliary disorders
- Gastrointestinal disorders
- Nervous system disorders
- Cardiac disorders
- Skin disorders
- And 22 more...

### Interpretation

| Categories Positive | Risk Level | Action |
|---------------------|------------|--------|
| 0-10 | Low | Monitor normally |
| 10-15 | Moderate | Consider in safety assessment |
| 15-20 | Elevated | May need optimization |
| > 20 | High | Significant safety concerns |

### Important Notes
- These are predictions based on structural similarity
- Not all predicted side effects will occur clinically
- Use as guidance for further testing priorities

---

## Diversity (Tanimoto Similarity)

### What It Measures
Structural similarity between molecules using Morgan fingerprints (radius=2).

### Interpretation

| Similarity | Relationship | Action |
|------------|--------------|--------|
| 1.0 | Identical | Same compound |
| 0.7-1.0 | Very similar | Likely similar activity |
| 0.5-0.7 | Similar | May have different properties |
| 0.3-0.5 | Different | Good diversity |
| < 0.3 | Very different | High diversity |

### Recommendations
- For lead selection: Choose compounds with similarity ≤ 0.7
- For SAR studies: Use similarity 0.5-0.8 range
- For scaffold hopping: Look for similarity < 0.5

---

## Combined Property Assessment

### Ideal Lead Candidate Profile

| Property | Target Range | Notes |
|----------|--------------|-------|
| Docking | < -10 kcal/mol | Strong binding |
| QED | > 0.5 | Good drug-likeness |
| SA | < 5 | Synthesizable |
| LogP | 1-5 | Balanced lipophilicity |
| Lipinski | ≥ 3 rules | Acceptable violations |
| BBB | Depends on target | CNS vs peripheral |
| Side Effects | < 15 categories | Lower risk |

### Trade-offs to Consider

1. **Potency vs Drug-likeness**: Higher potency often means larger molecules with more violations
2. **Potency vs Selectivity**: Strong binding may increase off-target effects
3. **Novelty vs Synthetic Accessibility**: Novel scaffolds are harder to synthesize
4. **CNS Activity vs Side Effects**: BBB penetration may increase CNS side effects
