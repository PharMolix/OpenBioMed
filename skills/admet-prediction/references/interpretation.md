# ADMET Interpretation Guidelines

This document provides detailed interpretation guidelines for all ADMET properties.

## 1. Blood-Brain Barrier Penetration (BBBP)

### What It Measures
The ability of a molecule to cross the blood-brain barrier and enter the central nervous system.

### Output Format
- **Type**: Classification probability (0-1)
- **Interpretation**: Probability that the molecule will cross BBB

### Interpretation Scale

| Value | Classification | Drug Development Implications |
|-------|----------------|-------------------------------|
| > 0.7 | High penetration | Good for CNS targets, risk of CNS side effects |
| 0.5 - 0.7 | Moderate penetration | May reach CNS at therapeutic doses |
| 0.3 - 0.5 | Low penetration | Limited CNS exposure |
| < 0.3 | Very low/no penetration | Suitable for peripheral targets only |

### Drug Development Considerations
- **CNS drugs**: Want BBBP > 0.5 (antidepressants, antipsychotics, anti-epileptics)
- **Peripheral drugs**: Want BBBP < 0.3 to avoid CNS side effects
- **Example**: Aspirin (0.19) - minimal CNS penetration, suitable for peripheral use

---

## 2. SIDER Side Effects

### What It Measures
Probability of a molecule causing side effects across 27 organ system categories.

### Output Format
- **Type**: Multi-label classification probabilities (27 values, 0-1 each)
- **Interpretation**: Higher values indicate higher risk of that side effect

### Risk Stratification

| Probability | Risk Level | Clinical Implication |
|-------------|------------|----------------------|
| > 0.7 | HIGH | Significant risk - consider structural modifications |
| 0.5 - 0.7 | MODERATE | Monitor in clinical trials |
| 0.3 - 0.5 | LOW | Generally acceptable |
| < 0.3 | MINIMAL | Unlikely to cause this side effect |

### Common High-Risk Categories for Drug Candidates

| Category | Common Culprits | Mitigation Strategies |
|----------|-----------------|----------------------|
| Gastrointestinal | NSAIDs, antibiotics | Formulation, dosing with food |
| Nervous system | CNS drugs, lipophilic compounds | Reduce BBB penetration |
| Skin disorders | Sulfonamides, penicillins | Structure modification |
| Hepatobiliary | Acetaminophen, statins | Hepatotoxicity monitoring |

---

## 3. Caco-2 Permeability

### What It Measures
Permeability across Caco-2 cell monolayer (model of intestinal absorption).

### Output Format
- **Type**: Regression (log cm/s)
- **Interpretation**: Higher (less negative) values indicate better absorption

### Interpretation Scale

| Log Papp (cm/s) | Papp (cm/s) | Absorption | Oral Bioavailability |
|-----------------|-------------|------------|---------------------|
| > -5 | > 10^-5 | High | Good (>70%) |
| -6 to -5 | 10^-6 to 10^-5 | Moderate | Fair (30-70%) |
| < -6 | < 10^-6 | Low | Poor (<30%) |

### Drug Development Implications
- **Oral drugs**: Target > -6 log cm/s
- **Low permeability compounds**: Consider prodrugs or formulation strategies
- **Alternative routes**: IV, transdermal for very low permeability

---

## 4. Half-Life (half_life_obach)

### What It Measures
The elimination half-life of the drug in hours.

### Output Format
- **Type**: Regression (log hours)
- **Interpretation**: Convert from log scale: t½ = 10^(output)

### Interpretation Scale

| Log Half-Life | Half-Life | Dosing Frequency |
|---------------|-----------|------------------|
| > 1.4 | > 25 h | Once daily or less |
| 0.7 - 1.4 | 5-25 h | Once to twice daily |
| 0 - 0.7 | 1-5 h | 2-4 times daily |
| < 0 | < 1 h | Frequent dosing or IV infusion |

### Drug Development Implications
- **Ideal range**: Log t½ 0.7-1.4 (5-25 hours) for once/twice daily dosing
- **Very short half-life**: Consider sustained-release formulations
- **Very long half-life**: Risk of accumulation, drug interactions

---

## 5. LD50 Toxicity (ld50_zhu)

### What It Measures
The median lethal dose - the dose that kills 50% of test animals.

### Output Format
- **Type**: Regression (log mg/kg)
- **Interpretation**: Convert from log scale: LD50 = 10^(output) mg/kg

### Toxicity Classification

| Log LD50 | LD50 (mg/kg) | GHS Category | Labeling |
|----------|--------------|--------------|----------|
| < 0 | < 1 | Category 1 | Fatal if swallowed |
| 0 - 1 | 1 - 10 | Category 2 | Fatal if swallowed |
| 1 - 2 | 10 - 100 | Category 3 | Toxic if swallowed |
| 2 - 3 | 100 - 1000 | Category 4 | Harmful if swallowed |
| > 3 | > 1000 | Category 5 | May be harmful |

### Drug Development Implications
- **Therapeutic index**: Compare LD50 to therapeutic dose
- **Safety margin**: Higher LD50 preferred
- **Most drugs**: LD50 > 100 mg/kg (log > 2) is typical

---

## Combined ADMET Assessment

### Ideal Drug Candidate Profile

| Property | Target Range | Rationale |
|----------|--------------|-----------|
| BBBP | < 0.3 (peripheral) or > 0.5 (CNS) | Match to target location |
| SIDER | No categories > 0.7 | Minimize side effect risk |
| Caco-2 | > -6 log cm/s | Adequate oral absorption |
| Half-life | 0.7-1.4 log h | Convenient dosing |
| LD50 | > 2 log mg/kg | Low acute toxicity |

### Red Flags

- **BBBP > 0.5** for non-CNS targets (unwanted CNS effects)
- **Any SIDER > 0.8** (high side effect risk)
- **Caco-2 < -7** (very poor oral absorption)
- **LD50 < 1** (high acute toxicity)

### Decision Framework

```
1. Is BBBP appropriate for target?
   ├── CNS target → Need BBBP > 0.5
   └── Peripheral target → Want BBBP < 0.3

2. Check top 3 SIDER risks
   └── If > 0.7, consider structural modifications

3. Is oral absorption viable?
   └── Caco-2 > -6 for oral route

4. Is dosing feasible?
   └── Half-life 0.7-1.4 log h for BID dosing

5. Is safety margin adequate?
   └── LD50 > 2 with therapeutic dose << LD50
```
