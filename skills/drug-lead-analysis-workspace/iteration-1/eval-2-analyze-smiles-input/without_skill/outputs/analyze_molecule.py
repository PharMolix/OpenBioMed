"""
Drug-likeness Analysis Script for Molecule: CC(C)Cc1ccc(cc1)C(C)C(=O)O
"""

import sys
sys.path.insert(0, '/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev')

from open_biomed.data import Molecule
from open_biomed.data.molecule import (
    MoleculeQEDTool, MoleculeSATool, MoleculeLogPTool, MoleculeLipinskiTool,
    calc_sa_score
)
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os
from datetime import datetime

# SMILES for analysis
SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

def analyze_molecule(smiles: str):
    """Perform comprehensive drug-likeness analysis on a molecule."""

    # Create molecule
    molecule = Molecule.from_smiles(smiles)
    molecule._add_rdmol()

    # Basic molecular properties
    num_atoms = molecule.get_num_atoms()
    mol_weight = Descriptors.ExactMolWt(molecule.rdmol)
    num_h_donors = Lipinski.NumHDonors(molecule.rdmol)
    num_h_acceptors = Lipinski.NumHAcceptors(molecule.rdmol)
    num_rotatable_bonds = Lipinski.NumRotatableBonds(molecule.rdmol)
    tpsa = Descriptors.TPSA(molecule.rdmol)

    # QED Score
    qed_tool = MoleculeQEDTool()
    qed_scores, qed_msgs = qed_tool.run(molecule=molecule)
    qed_score = qed_scores[0]

    # SA Score
    sa_tool = MoleculeSATool()
    sa_scores, sa_msgs = sa_tool.run(molecule=molecule)
    sa_score = sa_scores[0]

    # LogP
    logp_tool = MoleculeLogPTool()
    logp_scores, logp_msgs = logp_tool.run(molecule=molecule)
    logp_score = logp_scores[0]

    # Lipinski
    lipinski_tool = MoleculeLipinskiTool()
    lipinski_scores, lipinski_msgs = lipinski_tool.run(molecule=molecule)
    lipinski_score = lipinski_scores[0]

    # Lipinski Rule Details
    rule1_mw = mol_weight < 500
    rule2_hbd = num_h_donors <= 5
    rule3_hba = num_h_acceptors <= 10
    rule4_logp = -2 <= logp_score <= 5
    lipinski_violations = 4 - sum([rule1_mw, rule2_hbd, rule3_hba, rule4_logp])

    # Generate report
    report = f"""# Drug-likeness Analysis Report

## Molecule Information

| Property | Value |
|----------|-------|
| **SMILES** | `{smiles}` |
| **Analysis Date** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |

## Molecular Structure Analysis

The molecule has the following characteristics:
- **Number of Atoms**: {num_atoms}
- **Molecular Weight**: {mol_weight:.2f} Da
- **SMILES Canonical Form**: {Chem.MolToSmiles(molecule.rdmol)}

## Drug-likeness Scores

### 1. QED Score (Quantitative Estimate of Drug-likeness)

| Metric | Value |
|--------|-------|
| **QED Score** | {qed_score:.4f} |
| **Assessment** | {'Excellent (High drug-likeness)' if qed_score > 0.7 else 'Good (Moderate drug-likeness)' if qed_score > 0.5 else 'Poor (Low drug-likeness)'} |

**Interpretation**: QED is a measure of drug-likeness based on the distribution of molecular properties in FDA-approved drugs. Scores range from 0 to 1, with higher values indicating greater drug-likeness.

### 2. SA Score (Synthetic Accessibility)

| Metric | Value |
|--------|-------|
| **SA Score** | {sa_score:.2f} |
| **Assessment** | {'Easy to synthesize' if sa_score < 3 else 'Moderate synthetic difficulty' if sa_score < 6 else 'Hard to synthesize'} |

**Interpretation**: SA score ranges from 1 (easiest to synthesize) to 10 (most difficult). Lower scores indicate molecules that are easier to synthesize.

### 3. LogP (Lipophilicity)

| Metric | Value |
|--------|-------|
| **LogP** | {logp_score:.2f} |
| **Assessment** | {'Optimal range (-0.4 to 5.6)' if -0.4 <= logp_score <= 5.6 else 'Outside optimal range'} |

**Interpretation**: LogP measures lipophilicity. Optimal values for drug candidates typically fall between -0.4 and 5.6. Values too low may indicate poor membrane permeability, while values too high may indicate poor solubility.

### 4. Lipinski's Rule of Five

| Rule | Criterion | Value | Status |
|------|-----------|-------|--------|
| Rule 1 | Molecular Weight < 500 Da | {mol_weight:.2f} Da | {'PASS' if rule1_mw else 'FAIL'} |
| Rule 2 | H-Bond Donors <= 5 | {num_h_donors} | {'PASS' if rule2_hbd else 'FAIL'} |
| Rule 3 | H-Bond Acceptors <= 10 | {num_h_acceptors} | {'PASS' if rule3_hba else 'FAIL'} |
| Rule 4 | LogP between -2 and 5 | {logp_score:.2f} | {'PASS' if rule4_logp else 'FAIL'} |
| **Total Violations** | | **{lipinski_violations}** | {'PASS' if lipinski_violations == 0 else 'ACCEPTABLE' if lipinski_violations == 1 else 'CONCERN'} |

**Interpretation**: Compounds with 0 violations have excellent oral bioavailability potential. One violation is generally acceptable, while 2 or more violations may indicate poor oral bioavailability.

## Additional Molecular Properties

| Property | Value |
|----------|-------|
| **TPSA (Topological Polar Surface Area)** | {tpsa:.2f} A^2 |
| **Rotatable Bonds** | {num_rotatable_bonds} |
| **H-Bond Donors** | {num_h_donors} |
| **H-Bond Acceptors** | {num_h_acceptors} |

## Summary Assessment

| Metric | Value | Interpretation |
|--------|-------|----------------|
| QED Score | {qed_score:.4f} | {'Excellent drug-likeness' if qed_score > 0.7 else 'Good drug-likeness' if qed_score > 0.5 else 'Poor drug-likeness'} |
| SA Score | {sa_score:.2f} | {'Easy synthesis' if sa_score < 3 else 'Moderate synthesis' if sa_score < 6 else 'Hard synthesis'} |
| LogP | {logp_score:.2f} | {'Optimal lipophilicity' if -0.4 <= logp_score <= 5.6 else 'Suboptimal lipophilicity'} |
| Lipinski Violations | {lipinski_violations} | {'Full compliance' if lipinski_violations == 0 else 'Minor concern' if lipinski_violations == 1 else 'Major concern'} |

## Overall Drug Potential Assessment

This molecule shows **{'EXCELLENT' if qed_score > 0.7 and lipinski_violations == 0 and sa_score < 3 else 'GOOD' if qed_score > 0.5 and lipinski_violations <= 1 and sa_score < 6 else 'MODERATE'}** potential as a drug candidate based on the following criteria:

1. **Drug-likeness (QED)**: The molecule {'has excellent drug-likeness properties' if qed_score > 0.7 else 'has acceptable drug-likeness properties' if qed_score > 0.5 else 'has limited drug-likeness properties'} with a QED score of {qed_score:.4f}.

2. **Synthetic Accessibility**: With an SA score of {sa_score:.2f}, this molecule is {'straightforward to synthesize' if sa_score < 3 else 'has moderate synthetic challenges' if sa_score < 6 else 'presents significant synthetic challenges'}.

3. **Lipophilicity**: The LogP of {logp_score:.2f} is {'within the optimal range for drug candidates' if -0.4 <= logp_score <= 5.6 else 'outside the typical optimal range'}, suggesting {'good' if -0.4 <= logp_score <= 5.6 else 'potentially limited'} membrane permeability and solubility characteristics.

4. **Oral Bioavailability**: The molecule {'fully complies' if lipinski_violations == 0 else 'has minor violations' if lipinski_violations == 1 else 'has significant violations'} with Lipinski's Rule of Five, indicating {'excellent' if lipinski_violations == 0 else 'acceptable' if lipinski_violations == 1 else 'limited'} potential for oral bioavailability.

## Recommendations

{'This molecule is an excellent candidate for further drug development. Proceed with ADMET profiling and target identification studies.' if qed_score > 0.7 and lipinski_violations == 0 and sa_score < 3 else 'This molecule shows promise as a drug candidate. Consider structural optimization to improve drug-likeness scores and further evaluation of ADMET properties.' if qed_score > 0.5 and lipinski_violations <= 1 else 'This molecule may require structural modifications to improve its drug-likeness profile. Consider medicinal chemistry optimization before advancing to further development stages.'}

---
*Report generated using OpenBioMed drug-likeness analysis tools*
"""

    return report, {
        "qed": qed_score,
        "sa": sa_score,
        "logp": logp_score,
        "lipinski_violations": lipinski_violations,
        "mol_weight": mol_weight,
        "num_atoms": num_atoms,
        "h_donors": num_h_donors,
        "h_acceptors": num_h_acceptors,
        "tpsa": tpsa,
        "rotatable_bonds": num_rotatable_bonds
    }


if __name__ == "__main__":
    report, metrics = analyze_molecule(SMILES)

    # Save report
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(output_dir, "analysis_report.md")

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    print("\nKey Metrics:")
    print(f"  QED Score: {metrics['qed']:.4f}")
    print(f"  SA Score: {metrics['sa']:.2f}")
    print(f"  LogP: {metrics['logp']:.2f}")
    print(f"  Lipinski Violations: {metrics['lipinski_violations']}")
    print(f"  Molecular Weight: {metrics['mol_weight']:.2f} Da")
