"""
ADMET Prediction Example

This script demonstrates how to predict comprehensive ADMET properties
for a molecule using GraphMVP ensemble models.
"""

import os
os.environ["dataset_name"] = "BBBP"

import torch
torch.set_printoptions(precision=4, sci_mode=False)

from open_biomed.core.pipeline import InferencePipeline, EnsemblePipeline
from open_biomed.data import Molecule


# Output prompts for classification tasks
OUTPUT_PROMPTS = {
    "BBBP": "The blood-brain barrier penetration of the molecule is {output}",
    "SIDER": """The possibility of the molecule to exhibit the side effects are:
 Hepatobiliary disorders: {output[0]:.4f}
 Metabolism and nutrition disorders: {output[1]:.4f}
 Product issues: {output[2]:.4f}
 Eye disorders: {output[3]:.4f}
 Investigations: {output[4]:.4f}
 Musculoskeletal and connective tissue disorders: {output[5]:.4f}
 Gastrointestinal disorders: {output[6]:.4f}
 Social circumstances: {output[7]:.4f}
 Immune system disorders: {output[8]:.4f}
 Reproductive system and breast disorders: {output[9]:.4f}
 Neoplasms benign, malignant and unspecified: {output[10]:.4f}
 General disorders and administration site conditions: {output[11]:.4f}
 Endocrine disorders: {output[12]:.4f}
 Surgical and medical procedures: {output[13]:.4f}
 Vascular disorders: {output[14]:.4f}
 Blood and lymphatic system disorders: {output[15]:.4f}
 Skin and subcutaneous tissue disorders: {output[16]:.4f}
 Congenital, familial and genetic disorders: {output[17]:.4f}
 Infections and infestations: {output[18]:.4f}
 Respiratory, thoracic and mediastinal disorders: {output[19]:.4f}
 Psychiatric disorders: {output[20]:.4f}
 Renal and urinary disorders: {output[21]:.4f}
 Pregnancy, puerperium and perinatal conditions: {output[22]:.4f}
 Ear and labyrinth disorders: {output[23]:.4f}
 Cardiac disorders: {output[24]:.4f}
 Nervous system disorders: {output[25]:.4f}
 Injury, poisoning and procedural complications: {output[26]:.4f}
""",
}

# Output prompts for regression tasks
OUTPUT_PROMPTS_REGRESSION = {
    "caco2_wang": "The rate of drug passing through the Caco-2 cells is {output}",
    "half_life_obach": "The half-life of the molecule is {output}",
    "ld50_zhu": "The most conservative dose of the molecule that can lead to lethal adverse effects is {output}",
}


def build_admet_pipeline(device="cuda:0"):
    """Build ensemble pipeline for ADMET prediction."""
    pipelines = {}

    # Classification tasks
    for task in OUTPUT_PROMPTS:
        pipelines[task] = InferencePipeline(
            task="molecule_property_prediction",
            model="graphmvp",
            model_ckpt=f"./checkpoints/server/graphmvp-{task}.ckpt",
            additional_config=f"./configs/dataset/{task.lower()}.yaml",
            device=device,
            output_prompt=OUTPUT_PROMPTS[task],
        )

    # Regression tasks
    for task in OUTPUT_PROMPTS_REGRESSION:
        pipelines[task] = InferencePipeline(
            task="molecule_property_prediction_regression",
            model="graphmvp_regression",
            model_ckpt=f"./checkpoints/server/graphmvp-{task}.ckpt",
            additional_config=f"./configs/dataset/{task.lower()}.yaml",
            device=device,
            output_prompt=OUTPUT_PROMPTS_REGRESSION[task],
        )

    return EnsemblePipeline(pipelines)


def predict_admet(smiles: str, device: str = "cuda:0"):
    """
    Predict ADMET properties for a molecule.

    Args:
        smiles: SMILES string of the molecule
        device: Device to run inference on ("cuda:0" or "cpu")

    Returns:
        Dictionary containing all ADMET predictions
    """
    # Load molecule
    molecule = Molecule.from_smiles(smiles)

    # Build pipeline
    pipeline = build_admet_pipeline(device=device)

    # Run predictions
    results = {}

    # BBB Penetration
    bbb_output = pipeline.run(molecule=molecule, task="BBBP")
    results["bbbp"] = float(bbb_output[1][0].strip("[]"))

    # Side Effects
    sider_output = pipeline.run(molecule=molecule, task="SIDER")
    results["sider"] = eval(sider_output[1][0])

    # Caco-2 Permeability
    caco2_output = pipeline.run(molecule=molecule, task="caco2_wang")
    results["caco2"] = float(caco2_output[1][0].strip("[]"))

    # Half-Life
    half_life_output = pipeline.run(molecule=molecule, task="half_life_obach")
    results["half_life"] = float(half_life_output[1][0].strip("[]"))

    # LD50
    ld50_output = pipeline.run(molecule=molecule, task="ld50_zhu")
    results["ld50"] = float(ld50_output[1][0].strip("[]"))

    return results


def format_admet_report(smiles: str, results: dict) -> str:
    """Format ADMET results as human-readable report."""
    report = []
    report.append("=" * 50)
    report.append("ADMET Prediction Report")
    report.append("=" * 50)
    report.append(f"SMILES: {smiles}")
    report.append("")

    # Absorption
    report.append("--- Absorption ---")
    caco2_val = results["caco2"]
    caco2_interp = "High" if caco2_val > -5 else "Moderate" if caco2_val > -6 else "Low"
    report.append(f"Caco-2 Permeability: {caco2_val:.2f} log cm/s ({caco2_interp})")
    report.append("")

    # Distribution
    report.append("--- Distribution ---")
    bbb_val = results["bbbp"]
    bbb_interp = "Likely" if bbb_val > 0.5 else "Unlikely"
    report.append(f"BBB Penetration: {bbb_val:.4f} ({bbb_interp})")
    report.append("")

    # Metabolism & Excretion
    report.append("--- Metabolism & Excretion ---")
    half_life_val = results["half_life"]
    half_life_hours = 10 ** half_life_val
    report.append(f"Half-Life: {half_life_val:.2f} log h (~{half_life_hours:.1f} h)")
    report.append("")

    # Toxicity
    report.append("--- Toxicity ---")
    ld50_val = results["ld50"]
    ld50_mgkg = 10 ** ld50_val
    if ld50_val < 1:
        tox_level = "Highly toxic"
    elif ld50_val < 2:
        tox_level = "Moderately toxic"
    elif ld50_val < 3:
        tox_level = "Slightly toxic"
    else:
        tox_level = "Low toxicity"
    report.append(f"LD50: {ld50_val:.2f} log mg/kg (~{ld50_mgkg:.1f} mg/kg, {tox_level})")
    report.append("")

    # Side Effects
    report.append("--- Top Side Effect Risks (SIDER) ---")
    sider_names = [
        "Hepatobiliary disorders", "Metabolism/nutrition disorders", "Product issues",
        "Eye disorders", "Investigations", "Musculoskeletal disorders",
        "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
        "Reproductive disorders", "Neoplasms", "General disorders",
        "Endocrine disorders", "Surgical procedures", "Vascular disorders",
        "Blood/lymphatic disorders", "Skin disorders", "Congenital disorders",
        "Infections", "Respiratory disorders", "Psychiatric disorders",
        "Renal/urinary disorders", "Pregnancy conditions", "Ear disorders",
        "Cardiac disorders", "Nervous system disorders", "Injury/poisoning"
    ]
    sider_scores = list(zip(sider_names, results["sider"]))
    sider_scores.sort(key=lambda x: x[1], reverse=True)

    for name, score in sider_scores[:5]:
        risk = "HIGH" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW"
        report.append(f"  {name}: {score:.2f} ({risk})")

    report.append("")
    report.append("=" * 50)

    return "\n".join(report)


if __name__ == "__main__":
    # Example: Predict ADMET for Aspirin
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"Predicting ADMET for: {smiles}")
    print()

    results = predict_admet(smiles)
    report = format_admet_report(smiles, results)
    print(report)
