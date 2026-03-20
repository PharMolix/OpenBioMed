"""
Example: Drug Lead Analysis for Aspirin

This example demonstrates the drug-lead-analysis skill workflow.
Run this to see how the skill analyzes a molecule for drug potential.
"""

from open_biomed.tools import TOOLS
from open_biomed.data import Molecule

def analyze_drug_lead(molecule_name=None, smiles=None):
    """
    Perform comprehensive drug lead analysis on a molecule.

    Args:
        molecule_name: Common name of the molecule (e.g., "aspirin")
        smiles: SMILES string if name is not available
    """
    # Step 1: Get the molecule
    print("=" * 60)
    print("DRUG LEAD ANALYSIS")
    print("=" * 60)

    if molecule_name:
        print(f"\n[1] Retrieving molecule: {molecule_name}")
        tool = TOOLS["molecule_name_request"]
        result, message = tool.run(name=molecule_name)
        molecule = result["molecule"]
        print(f"    {message}")
    elif smiles:
        print(f"\n[1] Creating molecule from SMILES: {smiles}")
        molecule = Molecule.from_smiles(smiles)
        print(f"    Molecule created successfully")
    else:
        raise ValueError("Provide either molecule_name or smiles")

    # Step 2: Calculate drug-likeness scores
    print("\n[2] Calculating Drug-likeness Scores")
    print("-" * 40)

    # QED
    qed_tool = TOOLS["molecule_qed"]
    qed_result, qed_msg = qed_tool.run(molecule=molecule)
    qed_score = qed_result.get("qed", qed_result)
    print(f"    QED Score: {qed_score:.3f}")
    print(f"    Assessment: {'Excellent' if qed_score > 0.7 else 'Good' if qed_score > 0.5 else 'Poor'} drug-likeness")

    # SA Score
    sa_tool = TOOLS["molecule_sa"]
    sa_result, sa_msg = sa_tool.run(molecule=molecule)
    sa_score = sa_result.get("sa", sa_result)
    print(f"    SA Score: {sa_score:.2f}")
    print(f"    Assessment: {'Easy' if sa_score < 3 else 'Moderate' if sa_score < 6 else 'Hard'} to synthesize")

    # LogP
    logp_tool = TOOLS["molecule_logp"]
    logp_result, logp_msg = logp_tool.run(molecule=molecule)
    logp_score = logp_result.get("logp", logp_result)
    print(f"    LogP: {logp_score:.2f}")
    print(f"    Assessment: {'Optimal' if -0.4 <= logp_score <= 5.6 else 'Outside optimal range'}")

    # Lipinski
    lipinski_tool = TOOLS["molecule_lipinski"]
    lipinski_result, lipinski_msg = lipinski_tool.run(molecule=molecule)
    violations = lipinski_result.get("violations", lipinski_result)
    print(f"    Lipinski Violations: {violations}")
    print(f"    Assessment: {'Pass' if violations == 0 else 'Acceptable' if violations == 1 else 'Concern'}")

    # Step 3: Predict ADMET properties
    print("\n[3] Predicting ADMET Properties")
    print("-" * 40)

    try:
        prop_tool = TOOLS["molecule_property_prediction"]

        # BBB Penetration
        print("    Predicting BBB penetration...")
        bbb_result, bbb_msg = prop_tool.run(
            molecule=molecule,
            dataset="bbbp",
            model="graphmvp"
        )
        bbb_pred = bbb_result.get("prediction", bbb_result)
        print(f"    BBB Penetration: {'Yes' if bbb_pred == 1 else 'No'}")

        # Side Effects (optional - can be slow)
        # sider_result, sider_msg = prop_tool.run(
        #     molecule=molecule,
        #     dataset="sider",
        #     model="graphmvp"
        # )
        # print(f"    Predicted Side Effects: {sider_result}")
    except Exception as e:
        print(f"    Note: Property prediction requires model checkpoints.")
        print(f"    Error: {e}")

    # Step 4: Visualize
    print("\n[4] Visualization")
    print("-" * 40)
    try:
        viz_tool = TOOLS["visualize_molecule"]
        viz_result, viz_msg = viz_tool.run(
            molecule=molecule,
            style="ball_stick",
            show_hydrogen=False
        )
        print(f"    {viz_msg}")
    except Exception as e:
        print(f"    Visualization: {e}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Value':<15} {'Assessment'}")
    print("-" * 60)
    print(f"{'QED Score':<25} {qed_score:<15.3f} {'Excellent' if qed_score > 0.7 else 'Good' if qed_score > 0.5 else 'Poor'}")
    print(f"{'SA Score':<25} {sa_score:<15.2f} {'Easy' if sa_score < 3 else 'Moderate' if sa_score < 6 else 'Hard'}")
    print(f"{'LogP':<25} {logp_score:<15.2f} {'Optimal' if -0.4 <= logp_score <= 5.6 else 'Suboptimal'}")
    print(f"{'Lipinski Violations':<25} {violations:<15} {'Pass' if violations == 0 else 'Review'}")

    print("\n[Analysis Complete]")

    return {
        "qed": qed_score,
        "sa": sa_score,
        "logp": logp_score,
        "lipinski_violations": violations
    }


if __name__ == "__main__":
    # Example 1: Analyze aspirin by name
    print("\n" + "#" * 60)
    print("# Example: Analyzing Aspirin")
    print("#" * 60)
    analyze_drug_lead(molecule_name="aspirin")

    # Example 2: Analyze a custom molecule by SMILES
    # print("\n" + "#" * 60)
    # print("# Example: Custom Molecule")
    # print("#" * 60)
    # analyze_drug_lead(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin SMILES
