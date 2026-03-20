# Advanced DDI Analysis

## CYP Enzyme Interactions

Cytochrome P450 enzymes are the most common cause of pharmacokinetic DDIs.

### Major CYP Enzymes

| Enzyme | Substrate Examples | Inhibitors | Inducers |
|--------|-------------------|------------|----------|
| **CYP3A4** | Midazolam, Simvastatin, Nifedipine | Ketoconazole, Ritonavir | Rifampin, Carbamazepine |
| **CYP2D6** | Metoprolol, Codeine, Fluoxetine | Fluoxetine, Paroxetine | Not typically induced |
| **CYP2C9** | Warfarin, Phenytoin, Ibuprofen | Fluconazole, Amiodarone | Rifampin |
| **CYP2C19** | Omeprazole, Clopidogrel | Fluconazole, Omeprazole | Rifampin |
| **CYP1A2** | Theophylline, Caffeine | Fluvoxamine | Smoking, Omeprazole |

### Detecting CYP Interactions

```python
def analyze_cyp_interaction(drug_info_a: dict, drug_info_b: dict) -> dict:
    """
    Analyze potential CYP-mediated interactions between two drugs.

    Returns information about:
    - Shared metabolic pathways
    - Inhibitor/substrate conflicts
    - Inducer/substrate conflicts
    """
    cyp_enzymes = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"]

    interactions = []

    for enzyme in cyp_enzymes:
        # Check if drug A affects enzyme and drug B is substrate
        a_inhibits = enzyme in drug_info_a.get("inhibitors", [])
        a_induces = enzyme in drug_info_a.get("inducers", [])
        b_substrate = enzyme in drug_info_b.get("substrates", [])

        if a_inhibits and b_substrate:
            interactions.append({
                "type": "inhibition",
                "enzyme": enzyme,
                "perpetrator": drug_info_a["name"],
                "victim": drug_info_b["name"],
                "effect": f"{drug_info_a['name']} may increase {drug_info_b['name']} levels via {enzyme} inhibition"
            })

        if a_induces and b_substrate:
            interactions.append({
                "type": "induction",
                "enzyme": enzyme,
                "perpetrator": drug_info_a["name"],
                "victim": drug_info_b["name"],
                "effect": f"{drug_info_a['name']} may decrease {drug_info_b['name']} levels via {enzyme} induction"
            })

    return interactions
```

## Pharmacodynamic Interactions

### Mechanism Types

1. **Additive/Synergistic**
   - Two drugs with similar effects → enhanced response
   - Example: Aspirin + Warfarin → increased bleeding risk

2. **Antagonistic**
   - Two drugs with opposing effects → reduced efficacy
   - Example: NSAID + Antihypertensive → reduced BP control

3. **Shared Target**
   - Both drugs bind same protein
   - Example: Ibuprofen + Aspirin → both inhibit COX

### Detection via KEGG

```python
def find_target_overlap(drug_a_targets: list, drug_b_targets: list) -> list:
    """Find shared protein targets between two drugs."""
    # Parse target strings from KEGG
    targets_a = set()
    for t in drug_a_targets:
        # Extract gene symbol: "PTGS1 (COX1) [HSA:5742]"
        if "(" in t:
            gene = t.split("(")[0].strip()
            targets_a.add(gene)

    targets_b = set()
    for t in drug_b_targets:
        if "(" in t:
            gene = t.split("(")[0].strip()
            targets_b.add(gene)

    return list(targets_a & targets_b)
```

## Risk Stratification

### High-Risk Drug Classes

| Drug Class | Risk | Common DDIs |
|------------|------|-------------|
| **Anticoagulants** | Very High | Bleeding with antiplatelets, NSAIDs, SSRIs |
| **Antiarrhythmics** | High | QT prolongation with many drugs |
| **Anticonvulsants** | High | CYP induction affects many drugs |
| **Immunosuppressants** | High | Narrow therapeutic index |
| **HIV Protease Inhibitors** | High | CYP3A4 interactions |

### Clinical Decision Support

```python
def get_clinical_recommendation(interaction: dict) -> str:
    """Generate clinical recommendation based on interaction severity."""

    severity = interaction.get("severity", "")

    if severity == "Contraindicated":
        return (
            "DO NOT USE TOGETHER. "
            "Consider alternative therapy. "
            "If already prescribed, consult specialist immediately."
        )
    elif severity == "Precaution":
        return (
            "MONITOR CLOSELY. "
            "Consider dose adjustment. "
            "Monitor for adverse effects. "
            "Consider timing separation if applicable."
        )
    elif severity == "Caution":
        return (
            "BE AWARE. "
            "Routine monitoring recommended. "
            "Educate patient on warning signs."
        )
    else:
        return "No special action required."
```

## Batch Analysis for Polypharmacy

```python
def analyze_polypharmacy(drugs: list, analyzer) -> dict:
    """
    Comprehensive analysis for patients on multiple medications.

    Returns:
    - All pairwise interactions
    - Network analysis (highly connected drugs)
    - Risk score
    - Recommendations
    """
    result = analyzer.analyze(drugs)

    # Find drugs with most interactions
    interaction_counts = {}
    for interaction in result.get("interactions", []):
        for drug in [interaction["drug_a"], interaction["drug_b"]]:
            interaction_counts[drug] = interaction_counts.get(drug, 0) + 1

    # Sort by interaction count
    high_risk_drugs = sorted(
        interaction_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # Calculate risk score
    severity_weights = {"Contraindicated": 10, "Precaution": 3, "Caution": 1}
    risk_score = sum(
        severity_weights.get(i.get("severity", ""), 0)
        for i in result.get("interactions", [])
    )

    return {
        **result,
        "high_risk_drugs": high_risk_drugs,
        "risk_score": risk_score,
        "risk_level": "High" if risk_score > 10 else "Moderate" if risk_score > 3 else "Low"
    }
```

## SMILES-Based Prediction

For novel compounds not in KEGG, use structural similarity:

```python
from open_biomed.data import Molecule

def predict_ddi_by_similarity(smiles_a: str, smiles_b: str, known_ddis: dict) -> float:
    """
    Predict DDI likelihood based on structural similarity to known interactors.

    Args:
        smiles_a: SMILES of compound A
        smiles_b: SMILES of compound B
        known_ddis: Dictionary of known DDI pairs with their fingerprints

    Returns:
        Probability of interaction (0-1)
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    # Generate fingerprints
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)

    # Calculate similarity
    similarity = DataStructs.TanimotoSimilarity(fp_a, fp_b)

    # If structurally similar to known interactors, higher risk
    # This is a simplified heuristic
    return similarity
```
