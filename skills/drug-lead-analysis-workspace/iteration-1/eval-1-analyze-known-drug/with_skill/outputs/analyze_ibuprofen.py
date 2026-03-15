#!/usr/bin/env python
"""
Drug Lead Analysis for Ibuprofen
Following the workflow defined in SKILL.md
"""

import json
import sys
sys.path.insert(0, '/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev')

from open_biomed.tools import TOOLS
from open_biomed.data import Molecule

def main():
    results = {}

    print("=" * 60)
    print("DRUG LEAD ANALYSIS: IBUPROFEN")
    print("=" * 60)

    # Step 1: Get the Molecule (using molecule name)
    print("\n[Step 1] Retrieving molecule from PubChem...")
    try:
        tool = TOOLS["molecule_name_request"]
        result, message = tool.run(name="ibuprofen")
        molecule = result["molecule"]
        print(f"Retrieved: {message}")
        results["molecule_retrieval"] = {
            "status": "success",
            "message": message
        }
    except Exception as e:
        print(f"Error retrieving molecule: {e}")
        results["molecule_retrieval"] = {
            "status": "error",
            "error": str(e)
        }
        return results

    # Get SMILES for reference
    try:
        smiles = molecule.get_smiles()
        print(f"SMILES: {smiles}")
        results["smiles"] = smiles
    except Exception as e:
        print(f"Could not get SMILES: {e}")

    # Step 2: Calculate Drug-likeness Scores
    print("\n[Step 2] Calculating Drug-likeness Scores...")

    # QED Score
    try:
        qed_tool = TOOLS["molecule_qed"]
        qed_result, qed_msg = qed_tool.run(molecule=molecule)
        print(f"QED: {qed_msg}")
        results["qed"] = {
            "message": qed_msg,
            "result": qed_result
        }
    except Exception as e:
        print(f"QED calculation error: {e}")
        results["qed"] = {"error": str(e)}

    # SA Score
    try:
        sa_tool = TOOLS["molecule_sa"]
        sa_result, sa_msg = sa_tool.run(molecule=molecule)
        print(f"SA Score: {sa_msg}")
        results["sa"] = {
            "message": sa_msg,
            "result": sa_result
        }
    except Exception as e:
        print(f"SA calculation error: {e}")
        results["sa"] = {"error": str(e)}

    # LogP
    try:
        logp_tool = TOOLS["molecule_logp"]
        logp_result, logp_msg = logp_tool.run(molecule=molecule)
        print(f"LogP: {logp_msg}")
        results["logp"] = {
            "message": logp_msg,
            "result": logp_result
        }
    except Exception as e:
        print(f"LogP calculation error: {e}")
        results["logp"] = {"error": str(e)}

    # Lipinski's Rule of Five
    try:
        lipinski_tool = TOOLS["molecule_lipinski"]
        lipinski_result, lipinski_msg = lipinski_tool.run(molecule=molecule)
        print(f"Lipinski: {lipinski_msg}")
        results["lipinski"] = {
            "message": lipinski_msg,
            "result": lipinski_result
        }
    except Exception as e:
        print(f"Lipinski calculation error: {e}")
        results["lipinski"] = {"error": str(e)}

    # Step 3: Predict ADMET Properties
    print("\n[Step 3] Predicting ADMET Properties...")

    # Blood-brain barrier penetration
    try:
        prop_tool = TOOLS["molecule_property_prediction"]
        bbb_result, bbb_msg = prop_tool.run(
            molecule=molecule,
            dataset="bbbp",
            model="graphmvp"
        )
        print(f"BBB Penetration: {bbb_msg}")
        results["bbb"] = {
            "message": bbb_msg,
            "result": bbb_result
        }
    except Exception as e:
        print(f"BBB prediction error: {e}")
        results["bbb"] = {"error": str(e)}

    # Side effects prediction
    try:
        prop_tool = TOOLS["molecule_property_prediction"]
        sidefx_result, sidefx_msg = prop_tool.run(
            molecule=molecule,
            dataset="sider",
            model="graphmvp"
        )
        print(f"Side Effects: {sidefx_msg}")
        results["side_effects"] = {
            "message": sidefx_msg,
            "result": sidefx_result
        }
    except Exception as e:
        print(f"Side effects prediction error: {e}")
        results["side_effects"] = {"error": str(e)}

    # Step 4: Visualize the Molecule
    print("\n[Step 4] Visualizing the Molecule...")
    try:
        viz_tool = TOOLS["visualize_molecule"]
        viz_result, viz_msg = viz_tool.run(
            molecule=molecule,
            style="ball_stick",
            show_hydrogen=False
        )
        print(f"Visualization: {viz_msg}")
        results["visualization"] = {
            "message": viz_msg,
            "result": viz_result if isinstance(viz_result, str) else "generated"
        }
    except Exception as e:
        print(f"Visualization error: {e}")
        results["visualization"] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Save results to JSON for report generation
    with open('/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev/BioMedSkills/drug-lead-analysis-workspace/iteration-1/eval-1-analyze-known-drug/with_skill/outputs/analysis_results.json', 'w') as f:
        # Convert any non-serializable objects to strings
        def serialize(obj):
            if hasattr(obj, '__dict__'):
                return str(obj)
            elif isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            return str(obj)

        json.dump(results, f, default=serialize, indent=2)

    return results

if __name__ == "__main__":
    results = main()
