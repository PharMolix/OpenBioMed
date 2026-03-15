#!/usr/bin/env python
"""Analyze drug-likeness properties of aspirin and acetaminophen."""

import sys
sys.path.insert(0, '/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev')

from open_biomed.data import Molecule
from rdkit.Chem import Descriptors, Lipinski

# Create molecule objects
aspirin = Molecule.from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')
acetaminophen = Molecule.from_smiles('CC(=O)NC1=CC=C(O)C=C1')

# Calculate properties for aspirin
aspirin_props = {
    'name': 'Aspirin',
    'smiles': aspirin.smiles,
    'qed': aspirin.calc_qed(),
    'sa': aspirin.calc_sa(),
    'sa_norm': aspirin.calc_sa(normalize=True),
    'logp': aspirin.calc_logp(),
    'lipinski_passed': int(aspirin.calc_lipinski()),
    'mw': Descriptors.ExactMolWt(aspirin.rdmol),
    'hbd': Lipinski.NumHDonors(aspirin.rdmol),
    'hba': Lipinski.NumHAcceptors(aspirin.rdmol),
    'rotb': Lipinski.NumRotatableBonds(aspirin.rdmol),
    'atoms': aspirin.get_num_atoms(),
}

# Calculate properties for acetaminophen
acetaminophen_props = {
    'name': 'Acetaminophen',
    'smiles': acetaminophen.smiles,
    'qed': acetaminophen.calc_qed(),
    'sa': acetaminophen.calc_sa(),
    'sa_norm': acetaminophen.calc_sa(normalize=True),
    'logp': acetaminophen.calc_logp(),
    'lipinski_passed': int(acetaminophen.calc_lipinski()),
    'mw': Descriptors.ExactMolWt(acetaminophen.rdmol),
    'hbd': Lipinski.NumHDonors(acetaminophen.rdmol),
    'hba': Lipinski.NumHAcceptors(acetaminophen.rdmol),
    'rotb': Lipinski.NumRotatableBonds(acetaminophen.rdmol),
    'atoms': acetaminophen.get_num_atoms(),
}

# Calculate Lipinski violations details
def get_lipinski_details(props, mol):
    """Get detailed Lipinski rule violations."""
    mw_ok = props['mw'] < 500
    hbd_ok = props['hbd'] <= 5
    hba_ok = props['hba'] <= 10
    logp_ok = -2 <= props['logp'] <= 5

    violations = []
    if not mw_ok:
        violations.append(f"Molecular weight >= 500 Da ({props['mw']:.2f} Da)")
    if not hbd_ok:
        violations.append(f"H-bond donors > 5 ({props['hbd']})")
    if not hba_ok:
        violations.append(f"H-bond acceptors > 10 ({props['hba']})")
    if not logp_ok:
        violations.append(f"LogP outside [-2, 5] ({props['logp']:.2f})")

    return violations

aspirin_violations = get_lipinski_details(aspirin_props, aspirin)
acetaminophen_violations = get_lipinski_details(acetaminophen_props, acetaminophen)

# Generate comparison report
report = f"""# Drug Candidate Comparison Report: Aspirin vs Acetaminophen

## Executive Summary

This report provides a comprehensive analysis of drug-likeness properties for two lead candidates: **Aspirin** (acetylsalicylic acid) and **Acetaminophen** (paracetamol). The analysis focuses on key pharmaceutical properties relevant to lead optimization.

---

## Molecular Structures

| Property | Aspirin | Acetaminophen |
|----------|---------|---------------|
| **SMILES** | `{aspirin_props['smiles']}` | `{acetaminophen_props['smiles']}` |
| **Molecular Weight** | {aspirin_props['mw']:.2f} Da | {acetaminophen_props['mw']:.2f} Da |
| **Number of Atoms** | {aspirin_props['atoms']} | {acetaminophen_props['atoms']} |

---

## Drug-Likeness Properties

### Quantitative Estimate of Drug-likeness (QED) Score

The QED score measures how closely a molecule resembles known orally active drugs. Higher scores (0-1 scale) indicate better drug-likeness.

| Molecule | QED Score | Interpretation |
|----------|-----------|----------------|
| **Aspirin** | **{aspirin_props['qed']:.4f}** | {'Excellent' if aspirin_props['qed'] > 0.7 else 'Good' if aspirin_props['qed'] > 0.5 else 'Moderate'} |
| **Acetaminophen** | **{acetaminophen_props['qed']:.4f}** | {'Excellent' if acetaminophen_props['qed'] > 0.7 else 'Good' if acetaminophen_props['qed'] > 0.5 else 'Moderate'} |

**Winner: {('Aspirin' if aspirin_props['qed'] > acetaminophen_props['qed'] else 'Acetaminophen' if acetaminophen_props['qed'] > aspirin_props['qed'] else 'Tie')}** ({('aspirin by ' + f'{aspirin_props["qed"] - acetaminophen_props["qed"]:.4f}' if aspirin_props['qed'] > acetaminophen_props['qed'] else 'acetaminophen by ' + f'{acetaminophen_props["qed"] - aspirin_props["qed"]:.4f}' if acetaminophen_props['qed'] > aspirin_props['qed'] else 'equal scores')})

---

### Synthetic Accessibility (SA) Score

The SA score estimates how easy a molecule is to synthesize. Lower scores (1-10 scale) indicate easier synthesis.

| Molecule | SA Score | Normalized (0-1) | Interpretation |
|----------|----------|------------------|----------------|
| **Aspirin** | **{aspirin_props['sa']:.2f}** | {aspirin_props['sa_norm']:.2f} | {'Very Easy' if aspirin_props['sa'] < 3 else 'Easy' if aspirin_props['sa'] < 5 else 'Moderate'} |
| **Acetaminophen** | **{acetaminophen_props['sa']:.2f}** | {acetaminophen_props['sa_norm']:.2f} | {'Very Easy' if acetaminophen_props['sa'] < 3 else 'Easy' if acetaminophen_props['sa'] < 5 else 'Moderate'} |

**Winner: {('Aspirin' if aspirin_props['sa'] < acetaminophen_props['sa'] else 'Acetaminophen' if acetaminophen_props['sa'] < aspirin_props['sa'] else 'Tie')}** ({('aspirin with lower score' if aspirin_props['sa'] < acetaminophen_props['sa'] else 'acetaminophen with lower score' if acetaminophen_props['sa'] < aspirin_props['sa'] else 'equal scores')})

---

### Lipophilicity (LogP)

LogP measures the compound's distribution between octanol and water, indicating membrane permeability.

| Molecule | LogP | Interpretation |
|----------|------|----------------|
| **Aspirin** | **{aspirin_props['logp']:.2f}** | {'Optimal range' if -2 <= aspirin_props['logp'] <= 5 else 'Outside optimal range'} |
| **Acetaminophen** | **{acetaminophen_props['logp']:.2f}** | {'Optimal range' if -2 <= acetaminophen_props['logp'] <= 5 else 'Outside optimal range'} |

**Note:** Optimal LogP range for oral bioavailability is typically -2 to 5.

---

### Lipinski's Rule of Five

Lipinski's rules predict oral bioavailability. A compound should pass at least 3 of 4 rules.

| Rule | Threshold | Aspirin | Acetaminophen |
|------|-----------|---------|---------------|
| Molecular Weight | < 500 Da | {aspirin_props['mw']:.2f} Da {'PASS' if aspirin_props['mw'] < 500 else 'FAIL'} | {acetaminophen_props['mw']:.2f} Da {'PASS' if acetaminophen_props['mw'] < 500 else 'FAIL'} |
| H-bond Donors | <= 5 | {aspirin_props['hbd']} {'PASS' if aspirin_props['hbd'] <= 5 else 'FAIL'} | {acetaminophen_props['hbd']} {'PASS' if acetaminophen_props['hbd'] <= 5 else 'FAIL'} |
| H-bond Acceptors | <= 10 | {aspirin_props['hba']} {'PASS' if aspirin_props['hba'] <= 10 else 'FAIL'} | {acetaminophen_props['hba']} {'PASS' if acetaminophen_props['hba'] <= 10 else 'FAIL'} |
| LogP | -2 to 5 | {aspirin_props['logp']:.2f} {'PASS' if -2 <= aspirin_props['logp'] <= 5 else 'FAIL'} | {acetaminophen_props['logp']:.2f} {'PASS' if -2 <= acetaminophen_props['logp'] <= 5 else 'FAIL'} |
| **Total Passed** | - | **{aspirin_props['lipinski_passed']}/4** | **{acetaminophen_props['lipinski_passed']}/4** |

**Lipinski Violations:**
- **Aspirin:** {', '.join(aspirin_violations) if aspirin_violations else 'None - All rules passed'}
- **Acetaminophen:** {', '.join(acetaminophen_violations) if acetaminophen_violations else 'None - All rules passed'}

---

## Additional Molecular Properties

| Property | Aspirin | Acetaminophen |
|----------|---------|---------------|
| H-bond Donors | {aspirin_props['hbd']} | {acetaminophen_props['hbd']} |
| H-bond Acceptors | {aspirin_props['hba']} | {acetaminophen_props['hba']} |
| Rotatable Bonds | {aspirin_props['rotb']} | {acetaminophen_props['rotb']} |

---

## Comparative Summary

| Metric | Aspirin | Acetaminophen | Better |
|--------|---------|---------------|--------|
| QED Score | {aspirin_props['qed']:.4f} | {acetaminophen_props['qed']:.4f} | {'Aspirin' if aspirin_props['qed'] > acetaminophen_props['qed'] else 'Acetaminophen'} |
| SA Score | {aspirin_props['sa']:.2f} | {acetaminophen_props['sa']:.2f} | {'Aspirin' if aspirin_props['sa'] < acetaminophen_props['sa'] else 'Acetaminophen'} |
| LogP | {aspirin_props['logp']:.2f} | {acetaminophen_props['logp']:.2f} | {'Aspirin' if abs(aspirin_props['logp'] - 1.5) < abs(acetaminophen_props['logp'] - 1.5) else 'Acetaminophen'}* |
| Lipinski Rules | {aspirin_props['lipinski_passed']}/4 | {acetaminophen_props['lipinski_passed']}/4 | {'Aspirin' if aspirin_props['lipinski_passed'] > acetaminophen_props['lipinski_passed'] else 'Acetaminophen' if acetaminophen_props['lipinski_passed'] > aspirin_props['lipinski_passed'] else 'Tie'} |

*LogP comparison based on proximity to optimal value of ~1.5 for good oral bioavailability.

---

## Recommendation

Based on the comprehensive drug-likeness analysis:

### Winner: **{'Aspirin' if (aspirin_props['qed'] > acetaminophen_props['qed'] and aspirin_props['sa'] <= acetaminophen_props['sa']) or (aspirin_props['qed'] >= acetaminophen_props['qed'] and aspirin_props['sa'] < acetaminophen_props['sa']) else 'Acetaminophen' if (acetaminophen_props['qed'] > aspirin_props['qed'] and acetaminophen_props['sa'] <= aspirin_props['sa']) or (acetaminophen_props['qed'] >= aspirin_props['qed'] and acetaminophen_props['sa'] < aspirin_props['sa']) else 'Both are comparable'}**

### Key Findings:

1. **QED Score:** {'Aspirin' if aspirin_props['qed'] > acetaminophen_props['qed'] else 'Acetaminophen'} demonstrates better overall drug-likeness with a QED score of {max(aspirin_props['qed'], acetaminophen_props['qed']):.4f}.

2. **Synthetic Accessibility:** Both molecules are highly synthesizable with excellent SA scores. {'Aspirin' if aspirin_props['sa'] < acetaminophen_props['sa'] else 'Acetaminophen' if acetaminophen_props['sa'] < aspirin_props['sa'] else 'Both'} have{'s' if (aspirin_props['sa'] < acetaminophen_props['sa'] or acetaminophen_props['sa'] < aspirin_props['sa']) else ''} a slight advantage in synthesis.

3. **Lipinski Compliance:** Both molecules fully comply with Lipinski's Rule of Five, indicating good oral bioavailability potential.

4. **Lipophilicity:** Aspirin's LogP ({aspirin_props['logp']:.2f}) is {'higher' if aspirin_props['logp'] > acetaminophen_props['logp'] else 'lower'} than acetaminophen's ({acetaminophen_props['logp']:.2f}), suggesting {'better membrane permeability but potentially lower solubility' if aspirin_props['logp'] > acetaminophen_props['logp'] else 'better solubility but potentially lower membrane permeability'}.

### Final Recommendation for Lead Optimization:

For a lead optimization project, **{'aspirin' if aspirin_props['qed'] > acetaminophen_props['qed'] else 'acetaminophen'}** offers:
- {'Higher drug-likeness score' if max(aspirin_props['qed'], acetaminophen_props['qed']) == aspirin_props['qed'] else 'Better drug-likeness score'}
- Excellent synthetic accessibility
- Full Lipinski compliance
- {'Slightly better' if abs(aspirin_props['logp'] - acetaminophen_props['logp']) < 0.5 else 'Different'} lipophilicity profile

However, the choice should also consider:
- Target-specific requirements
- Desired mechanism of action (COX inhibition vs. analgesic/antipyretic)
- Toxicity profile considerations
- Desired pharmacokinetic properties

---

## Methodology

This analysis was performed using:
- **OpenBioMed** molecular property calculation tools
- **RDKit** cheminformatics library
- Standard drug-likeness metrics (QED, SA, LogP, Lipinski)

---

*Report generated on 2026-03-14*
"""

# Write report to file
output_path = '/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev/BioMedSkills/drug-lead-analysis-workspace/iteration-1/eval-3-compare-drug-candidates/without_skill/outputs/comparison_report.md'
with open(output_path, 'w') as f:
    f.write(report)

print(f"Report saved to: {output_path}")
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"\nAspirin:")
print(f"  QED: {aspirin_props['qed']:.4f}")
print(f"  SA:  {aspirin_props['sa']:.2f}")
print(f"  LogP: {aspirin_props['logp']:.2f}")
print(f"  Lipinski: {aspirin_props['lipinski_passed']}/4")

print(f"\nAcetaminophen:")
print(f"  QED: {acetaminophen_props['qed']:.4f}")
print(f"  SA:  {acetaminophen_props['sa']:.2f}")
print(f"  LogP: {acetaminophen_props['logp']:.2f}")
print(f"  Lipinski: {acetaminophen_props['lipinski_passed']}/4")

print(f"\nRECOMMENDATION: {'Aspirin' if aspirin_props['qed'] > acetaminophen_props['qed'] else 'Acetaminophen' if acetaminophen_props['qed'] > aspirin_props['qed'] else 'Both are comparable'} shows better overall drug-likeness properties.")
