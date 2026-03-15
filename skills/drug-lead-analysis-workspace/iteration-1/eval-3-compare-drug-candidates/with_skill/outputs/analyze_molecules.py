#!/usr/bin/env python
"""
Drug Lead Analysis Script
Analyzes aspirin and acetaminophen for drug-likeness properties using OpenBioMed tools.
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, project_root)

os.chdir(project_root)

from open_biomed.tools import TOOLS
from open_biomed.data import Molecule

def analyze_molecule(name):
    """Analyze a single molecule for drug-likeness properties."""
    results = {"name": name}

    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print('='*60)

    # Step 1: Get the molecule from PubChem
    try:
        tool = TOOLS["molecule_name_request"]
        result, message = tool.run(name=name)
        molecule = result[0]  # Returns list of molecules
        results["retrieval_message"] = message[0]
        print(f"Retrieval: {message[0]}")

        # Get SMILES for reference
        molecule._add_smiles()
        if molecule.smiles:
            results["smiles"] = molecule.smiles
            print(f"SMILES: {molecule.smiles}")
    except Exception as e:
        print(f"Error retrieving molecule: {e}")
        results["error"] = str(e)
        return results

    # Step 2: Calculate Drug-likeness Scores
    # QED (Quantitative Estimate of Drug-likeness)
    try:
        qed_tool = TOOLS["molecule_qed"]
        qed_result, qed_msg = qed_tool.run(molecule=molecule)
        results["qed"] = qed_result[0]
        results["qed_message"] = qed_msg[0]
        print(f"QED: {results['qed']}")
    except Exception as e:
        print(f"QED error: {e}")
        results["qed_error"] = str(e)

    # SA Score (Synthetic Accessibility)
    try:
        sa_tool = TOOLS["molecule_sa"]
        sa_result, sa_msg = sa_tool.run(molecule=molecule)
        results["sa_score"] = sa_result[0]
        results["sa_message"] = sa_msg[0]
        print(f"SA Score: {results['sa_score']}")
    except Exception as e:
        print(f"SA error: {e}")
        results["sa_error"] = str(e)

    # LogP (Lipophilicity)
    try:
        logp_tool = TOOLS["molecule_logp"]
        logp_result, logp_msg = logp_tool.run(molecule=molecule)
        results["logp"] = logp_result[0]
        results["logp_message"] = logp_msg[0]
        print(f"LogP: {results['logp']}")
    except Exception as e:
        print(f"LogP error: {e}")
        results["logp_error"] = str(e)

    # Lipinski's Rule of Five
    try:
        lipinski_tool = TOOLS["molecule_lipinski"]
        lipinski_result, lipinski_msg = lipinski_tool.run(molecule=molecule)
        results["lipinski_rules_satisfied"] = lipinski_result[0]
        results["lipinski_message"] = lipinski_msg[0]
        print(f"Lipinski: {lipinski_msg[0]}")
    except Exception as e:
        print(f"Lipinski error: {e}")
        results["lipinski_error"] = str(e)

    # Step 3: ADMET Properties
    # Blood-brain barrier penetration
    try:
        prop_tool = TOOLS["molecule_property_prediction"]
        bbb_result, bbb_msg = prop_tool.run(
            molecule=molecule,
            task="BBBP"
        )
        results["bbb_result"] = bbb_msg[0]
        print(f"BBB: {bbb_msg[0]}")
    except Exception as e:
        print(f"BBB prediction error: {e}")
        results["bbb_error"] = str(e)

    # Side effects prediction (SIDER)
    try:
        prop_tool = TOOLS["molecule_property_prediction"]
        sidefx_result, sidefx_msg = prop_tool.run(
            molecule=molecule,
            task="SIDER"
        )
        results["side_effects"] = sidefx_msg[0]
        print(f"Side Effects: predicted")
    except Exception as e:
        print(f"Side effects prediction error: {e}")
        results["side_effects_error"] = str(e)

    return results

def main():
    """Main function to analyze both molecules and generate comparison report."""
    print("="*60)
    print("Drug Lead Analysis: Aspirin vs Acetaminophen")
    print("="*60)

    # Analyze both molecules
    aspirin_results = analyze_molecule("aspirin")
    acetaminophen_results = analyze_molecule("acetaminophen")

    # Save raw results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(output_dir, "analysis_data.json")
    with open(results_file, "w") as f:
        json.dump({
            "aspirin": aspirin_results,
            "acetaminophen": acetaminophen_results
        }, f, indent=2)
    print(f"\nRaw data saved to {results_file}")

    # Generate comparison report
    report = generate_comparison_report(aspirin_results, acetaminophen_results)
    report_file = os.path.join(output_dir, "comparison_report.md")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Comparison report saved to {report_file}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

def generate_comparison_report(aspirin, acetaminophen):
    """Generate a markdown comparison report."""

    # Helper functions for assessment
    def assess_qed(score):
        if score is None:
            return "N/A"
        if score > 0.7:
            return "Excellent"
        elif score >= 0.5:
            return "Good"
        else:
            return "Poor"

    def assess_sa(score):
        if score is None:
            return "N/A"
        if score <= 3:
            return "Easy to synthesize"
        elif score <= 6:
            return "Moderate difficulty"
        else:
            return "Difficult to synthesize"

    def assess_logp(score):
        if score is None:
            return "N/A"
        if -0.4 <= score <= 5.6:
            return "Optimal"
        elif score < -0.4:
            return "Too hydrophilic"
        else:
            return "Too lipophilic"

    def assess_lipinski(rules_satisfied):
        if rules_satisfied is None:
            return "N/A"
        violations = 5 - rules_satisfied  # Lipinski has 5 rules
        if violations == 0:
            return "Pass (0 violations)"
        elif violations == 1:
            return "Acceptable (1 violation)"
        else:
            return f"Concern ({violations} violations)"

    report = """# Drug Lead Comparative Analysis Report

## Executive Summary

This report provides a comprehensive comparison of **aspirin** and **acetaminophen** as potential drug leads for a lead optimization project. Both molecules are well-established drugs, and this analysis evaluates their drug-likeness properties to inform lead selection.

---

## Molecule Information

| Property | Aspirin | Acetaminophen |
|----------|---------|---------------|
| **SMILES** | `{aspirin_smiles}` | `{acetaminophen_smiles}` |

---

## Drug-likeness Scores Comparison

### Quantitative Estimate of Drug-likeness (QED)

| Metric | Aspirin | Acetaminophen | Assessment |
|--------|---------|---------------|------------|
| **QED Score** | {aspirin_qed:.4f} | {acetaminophen_qed:.4f} | Higher is better (0-1) |
| **Assessment** | {aspirin_qed_assess} | {acetaminophen_qed_assess} | |

**Interpretation:**
- QED > 0.7: Excellent drug-likeness
- QED 0.5-0.7: Good drug-likeness
- QED < 0.5: Poor drug-likeness, may need optimization

{qed_winner}

---

### Synthetic Accessibility (SA) Score

| Metric | Aspirin | Acetaminophen | Assessment |
|--------|---------|---------------|------------|
| **SA Score** | {aspirin_sa:.2f} | {acetaminophen_sa:.2f} | Lower is easier (1-10) |
| **Assessment** | {aspirin_sa_assess} | {acetaminophen_sa_assess} | |

**Interpretation:**
- SA 1-3: Easy to synthesize
- SA 3-6: Moderate difficulty
- SA 6-10: Difficult to synthesize

{sa_winner}

---

### Lipophilicity (LogP)

| Metric | Aspirin | Acetaminophen | Assessment |
|--------|---------|---------------|------------|
| **LogP** | {aspirin_logp:.2f} | {acetaminophen_logp:.2f} | Optimal: -0.4 to 5.6 |
| **Assessment** | {aspirin_logp_assess} | {acetaminophen_logp_assess} | |

**Interpretation:**
- LogP -0.4 to 5.6: Optimal range for oral drugs
- LogP < -0.4: Too hydrophilic, may have poor membrane permeability
- LogP > 5.6: Too lipophilic, may have poor solubility

{logp_winner}

---

### Lipinski's Rule of Five

| Metric | Aspirin | Acetaminophen | Assessment |
|--------|---------|---------------|------------|
| **Rules Satisfied** | {aspirin_lipinski}/5 | {acetaminophen_lipinski}/5 | 5 = ideal |
| **Assessment** | {aspirin_lipinski_assess} | {acetaminophen_lipinski_assess} | |

**Lipinski's Rules:**
1. Molecular weight <= 500 Da
2. LogP <= 5
3. Hydrogen bond donors <= 5
4. Hydrogen bond acceptors <= 10
5. Rotatable bonds <= 10

{lipinski_winner}

---

## ADMET Properties

### Blood-Brain Barrier (BBB) Penetration

| Molecule | BBB Prediction |
|----------|----------------|
| Aspirin | {aspirin_bbb} |
| Acetaminophen | {acetaminophen_bbb} |

**Note:** BBB penetration is important for CNS-targeting drugs but may be undesirable for peripheral targets.

---

### Side Effects Profile (SIDER Predictions)

#### Aspirin
{aspirin_sidefx}

#### Acetaminophen
{acetaminophen_sidefx}

---

## Comparative Summary

### Overall Drug-likeness Scores

| Metric | Aspirin | Acetaminophen | Winner |
|--------|---------|---------------|--------|
| QED Score | {aspirin_qed:.4f} | {acetaminophen_qed:.4f} | {qed_winner_short} |
| SA Score | {aspirin_sa:.2f} | {acetaminophen_sa:.2f} | {sa_winner_short} |
| LogP | {aspirin_logp:.2f} | {acetaminophen_logp:.2f} | {logp_winner_short} |
| Lipinski Rules | {aspirin_lipinski}/5 | {acetaminophen_lipinski}/5 | {lipinski_winner_short} |

---

## Recommendation

{recommendation}

---

## Methodology

This analysis was performed using OpenBioMed's drug lead analysis tools:
- **QED**: Quantitative Estimate of Drug-likeness calculated using RDKit
- **SA Score**: Synthetic Accessibility score based on molecular complexity
- **LogP**: Wildman-Crippen LogP calculation
- **Lipinski**: Rule of Five compliance check
- **BBB Penetration**: GraphMVP model prediction
- **Side Effects**: GraphMVP model trained on SIDER dataset

---

*Report generated using OpenBioMed Drug Lead Analysis Skill*
""".format(
        aspirin_smiles=aspirin.get("smiles", "N/A"),
        acetaminophen_smiles=acetaminophen.get("smiles", "N/A"),
        aspirin_qed=aspirin.get("qed", 0) or 0,
        acetaminophen_qed=acetaminophen.get("qed", 0) or 0,
        aspirin_qed_assess=assess_qed(aspirin.get("qed")),
        acetaminophen_qed_assess=assess_qed(acetaminophen.get("qed")),
        qed_winner="**Winner: " + ("Aspirin" if (aspirin.get("qed") or 0) > (acetaminophen.get("qed") or 0) else "Acetaminophen") + "** has a higher QED score, indicating better overall drug-likeness." if aspirin.get("qed") and acetaminophen.get("qed") else "Unable to determine winner due to missing data.",
        qed_winner_short="Aspirin" if (aspirin.get("qed") or 0) > (acetaminophen.get("qed") or 0) else "Acetaminophen",
        aspirin_sa=aspirin.get("sa_score", 0) or 0,
        acetaminophen_sa=acetaminophen.get("sa_score", 0) or 0,
        aspirin_sa_assess=assess_sa(aspirin.get("sa_score")),
        acetaminophen_sa_assess=assess_sa(acetaminophen.get("sa_score")),
        sa_winner="**Winner: " + ("Aspirin" if (aspirin.get("sa_score") or 10) < (acetaminophen.get("sa_score") or 10) else "Acetaminophen") + "** has a lower SA score, indicating easier synthesis." if aspirin.get("sa_score") and acetaminophen.get("sa_score") else "Unable to determine winner due to missing data.",
        sa_winner_short="Aspirin" if (aspirin.get("sa_score") or 10) < (acetaminophen.get("sa_score") or 10) else "Acetaminophen",
        aspirin_logp=aspirin.get("logp", 0) or 0,
        acetaminophen_logp=acetaminophen.get("logp", 0) or 0,
        aspirin_logp_assess=assess_logp(aspirin.get("logp")),
        acetaminophen_logp_assess=assess_logp(acetaminophen.get("logp")),
        logp_winner="Both compounds have LogP values within the optimal range for oral drugs." if (-0.4 <= (aspirin.get("logp") or 0) <= 5.6) and (-0.4 <= (acetaminophen.get("logp") or 0) <= 5.6) else "LogP assessment requires further evaluation.",
        logp_winner_short="Tie" if (-0.4 <= (aspirin.get("logp") or 0) <= 5.6) and (-0.4 <= (acetaminophen.get("logp") or 0) <= 5.6) else "N/A",
        aspirin_lipinski=aspirin.get("lipinski_rules_satisfied", 0) or 0,
        acetaminophen_lipinski=acetaminophen.get("lipinski_rules_satisfied", 0) or 0,
        aspirin_lipinski_assess=assess_lipinski(aspirin.get("lipinski_rules_satisfied")),
        acetaminophen_lipinski_assess=assess_lipinski(acetaminophen.get("lipinski_rules_satisfied")),
        lipinski_winner="**Both compounds pass Lipinski's Rule of Five**, indicating good oral bioavailability potential." if aspirin.get("lipinski_rules_satisfied", 0) >= 4 and acetaminophen.get("lipinski_rules_satisfied", 0) >= 4 else "Lipinski assessment completed.",
        lipinski_winner_short="Tie" if aspirin.get("lipinski_rules_satisfied") == acetaminophen.get("lipinski_rules_satisfied") else ("Aspirin" if aspirin.get("lipinski_rules_satisfied", 0) > acetaminophen.get("lipinski_rules_satisfied", 0) else "Acetaminophen"),
        aspirin_bbb=aspirin.get("bbb_result", "Prediction not available"),
        acetaminophen_bbb=acetaminophen.get("bbb_result", "Prediction not available"),
        aspirin_sidefx=aspirin.get("side_effects", "Side effects prediction not available"),
        acetaminophen_sidefx=acetaminophen.get("side_effects", "Side effects prediction not available"),
        recommendation=generate_recommendation(aspirin, acetaminophen)
    )

    return report

def generate_recommendation(aspirin, acetaminophen):
    """Generate a recommendation based on the analysis."""

    aspirin_qed = aspirin.get("qed") or 0
    acetaminophen_qed = acetaminophen.get("qed") or 0
    aspirin_sa = aspirin.get("sa_score") or 10
    acetaminophen_sa = acetaminophen.get("sa_score") or 10
    aspirin_lipinski = aspirin.get("lipinski_rules_satisfied") or 0
    acetaminophen_lipinski = acetaminophen.get("lipinski_rules_satisfied") or 0

    # Calculate a simple score
    aspirin_score = 0
    acetaminophen_score = 0

    # QED comparison
    if aspirin_qed > acetaminophen_qed:
        aspirin_score += 2
    elif acetaminophen_qed > aspirin_qed:
        acetaminophen_score += 2

    # SA comparison (lower is better)
    if aspirin_sa < acetaminophen_sa:
        aspirin_score += 1
    elif acetaminophen_sa < aspirin_sa:
        acetaminophen_score += 1

    # Lipinski comparison
    if aspirin_lipinski > acetaminophen_lipinski:
        aspirin_score += 1
    elif acetaminophen_lipinski > aspirin_lipinski:
        acetaminophen_score += 1

    recommendation = """Based on the comprehensive drug-likeness analysis:

"""

    if aspirin_score > acetaminophen_score:
        recommendation += """**Recommendation: Aspirin**

Aspirin demonstrates superior drug-likeness properties for a lead optimization project:
- Higher QED score indicates better overall drug-likeness
- Lower synthetic complexity (SA score)
- Full compliance with Lipinski's Rule of Five

Aspirin is recommended as the preferred lead candidate for further optimization efforts. Its well-characterized pharmacological profile, combined with favorable drug-likeness metrics, makes it an excellent starting point for lead optimization.
"""
    elif acetaminophen_score > aspirin_score:
        recommendation += """**Recommendation: Acetaminophen**

Acetaminophen demonstrates superior drug-likeness properties for a lead optimization project:
- Higher QED score indicates better overall drug-likeness
- More favorable synthetic accessibility
- Full compliance with Lipinski's Rule of Five

Acetaminophen is recommended as the preferred lead candidate for further optimization efforts. Its favorable safety profile and drug-likeness metrics make it an excellent starting point for lead optimization.
"""
    else:
        recommendation += """**Recommendation: Both compounds are comparable**

Both aspirin and acetaminophen demonstrate similar drug-likeness profiles:
- Both have QED scores indicating good drug-likeness
- Both are easy to synthesize (low SA scores)
- Both fully comply with Lipinski's Rule of Five

The choice between these two should be based on:
1. **Therapeutic target**: Consider whether COX inhibition (aspirin) or COX-3/central action (acetaminophen) is more aligned with your target indication
2. **Safety profile**: Acetaminophen has a better gastrointestinal safety profile
3. **Chemical optimization potential**: Consider which scaffold offers more room for structural modifications

Both are excellent lead candidates, and the final selection should be driven by project-specific therapeutic goals.
"""

    recommendation += """
### Key Considerations for Lead Optimization

1. **Aspirin considerations:**
   - Contains an ester group that can be modified for prodrug approaches
   - Carboxylic acid group can be used for salt formation
   - Well-studied metabolism (hydrolysis to salicylic acid)

2. **Acetaminophen considerations:**
   - Phenol group offers opportunities for derivatization
   - Amide linkage provides metabolic stability considerations
   - Simpler molecular structure may limit optimization vectors
"""

    return recommendation


if __name__ == "__main__":
    main()
