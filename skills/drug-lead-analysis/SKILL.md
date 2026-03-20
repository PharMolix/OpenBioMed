---
name: drug-lead-analysis
description: >
  Analyze drug candidate molecules for drug-likeness, ADMET properties, and safety profiles.
  Use this skill when:
  (1) Evaluating a molecule's potential as a drug candidate,
  (2) Checking drug-likeness scores (QED, Lipinski),
  (3) Predicting blood-brain barrier penetration,
  (4) Assessing side effects and ADMET properties,
  (5) Comparing multiple molecules for lead optimization.
license: MIT
category: drug-discovery
tags: [admet, drug-likeness, lead-optimization, molecular-properties]
---

# Drug Lead Analysis

This skill guides you through a comprehensive analysis of drug candidate molecules using OpenBioMed's molecular analysis tools.

## When to Use This Skill

- User asks to analyze a molecule for drug potential
- User provides a molecule name or SMILES and wants an evaluation
- User asks about drug-likeness, ADMET, BBB penetration, or side effects
- User wants to compare multiple molecules for lead optimization

## Analysis Workflow

### Step 1: Get the Molecule

First, obtain the molecule object:

**If user provides a molecule name** (e.g., "aspirin", "ibuprofen"):
```python
from open_biomed.tools import TOOLS

tool = TOOLS["molecule_name_request"]
result, message = tool.run(name="aspirin")
molecule = result["molecule"]
print(message)  # Shows retrieved info
```

**If user provides a SMILES string**:
```python
from open_biomed.data import Molecule

molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
```

**If user provides a SDF file**:
```python
molecule = Molecule.from_sdf_file("path/to/molecule.sdf")
```

### Step 2: Calculate Drug-likeness Scores

Run all drug-likeness metrics:

```python
from open_biomed.tools import TOOLS

# QED (Quantitative Estimate of Drug-likeness) - 0 to 1, higher is better
qed_tool = TOOLS["molecule_qed"]
qed_result, qed_msg = qed_tool.run(molecule=molecule)

# SA (Synthetic Accessibility) - 1 to 10, lower is easier to synthesize
sa_tool = TOOLS["molecule_sa"]
sa_result, sa_msg = sa_tool.run(molecule=molecule)

# LogP (lipophilicity) - ideally between -0.4 and 5.6
logp_tool = TOOLS["molecule_logp"]
logp_result, logp_msg = logp_tool.run(molecule=molecule)

# Lipinski's Rule of Five - count violations (0 is ideal)
lipinski_tool = TOOLS["molecule_lipinski"]
lipinski_result, lipinski_msg = lipinski_tool.run(molecule=molecule)
```

### Step 3: Predict ADMET Properties

Use the property prediction models:

```python
# Blood-brain barrier penetration (binary: penetrates or not)
prop_tool = TOOLS["molecule_property_prediction"]
bbb_result, bbb_msg = prop_tool.run(
    molecule=molecule,
    dataset="bbbp",
    model="graphmvp"
)

# Side effects prediction (27 categories from SIDER dataset)
sidefx_result, sidefx_msg = prop_tool.run(
    molecule=molecule,
    dataset="sider",
    model="graphmvp"
)
```

### Step 4: Visualize the Molecule

```python
viz_tool = TOOLS["visualize_molecule"]
viz_result, viz_msg = viz_tool.run(
    molecule=molecule,
    style="ball_stick",  # Options: "ball_stick", "stick", "line", "sphere"
    show_hydrogen=False
)
```

### Step 5: Summarize Findings

Present a structured report:

```
## Drug Lead Analysis Report: [Molecule Name]

### Drug-likeness Scores
| Metric | Value | Assessment |
|--------|-------|------------|
| QED | X.XX | [Good/Moderate/Poor] |
| SA Score | X.X | [Easy/Moderate/Hard to synthesize] |
| LogP | X.XX | [Optimal/High/Low] |
| Lipinski Violations | X | [Pass/Concern] |

### ADMET Properties
- Blood-Brain Barrier: [Penetrates/Does not penetrate]
- Predicted Side Effects: [List any predicted]

### Overall Assessment
[Summary of drug potential and recommendations]
```

## Interpretation Guidelines

### QED Score
- **> 0.7**: Excellent drug-likeness
- **0.5 - 0.7**: Good drug-likeness
- **< 0.5**: Poor drug-likeness, may need optimization

### SA Score (Synthetic Accessibility)
- **1-3**: Easy to synthesize
- **3-6**: Moderate difficulty
- **6-10**: Difficult to synthesize

### LogP (Lipophilicity)
- **-0.4 to 5.6**: Optimal range for oral drugs
- **< -0.4**: Too hydrophilic, may have poor membrane permeability
- **> 5.6**: Too lipophilic, may have poor solubility

### Lipinski's Rule of Five
A "drug-like" molecule should have:
- Molecular weight ≤ 500 Da
- LogP ≤ 5
- Hydrogen bond donors ≤ 5
- Hydrogen bond acceptors ≤ 10

Violations: 0 = ideal, 1 = acceptable, 2+ = concerning

## Example Usage

**User**: "Analyze aspirin as a drug candidate"

**Response workflow**:
1. Retrieve aspirin from PubChem
2. Calculate QED, SA, LogP, Lipinski scores
3. Predict BBB penetration and side effects
4. Visualize the molecule
5. Generate comprehensive report

## Error Handling

- If molecule name is not found in PubChem, ask user for SMILES or SDF file
- If property prediction fails, still provide drug-likeness scores
- Always explain what each metric means in context
