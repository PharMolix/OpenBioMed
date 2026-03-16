#!/usr/bin/env python3
import sys
import argparse
import json
import logging
try:
    import requests
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_smiles_from_pubchem(name):
    # Retrieve canonical SMILES from PubChem by common name/IUPAC
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("CanonicalSMILES")
    except Exception as e:
        pass
    return None

def analyze_smiles(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
    except ImportError:
        return {"error": "RDKit is not installed. Please run `pip install rdkit`."}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"error": f"Invalid SMILES string: {smiles}"}
        
        # Core property calculations
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        
        # Lipinski Rule Break Check
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        analysis = {
            "Input_SMILES": smiles,
            "Canonical_SMILES": canonical_smiles,
            "Physicochemical_Properties": {
                "MolWt": round(mw, 2),
                "ExactMass": round(Descriptors.ExactMolWt(mol), 4),
                "LogP": round(logp, 2),
                "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2)
            },
            "Structural_Features": {
                "NumAtoms": mol.GetNumAtoms(),
                "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
                "NumBonds": mol.GetNumBonds(),
                "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
                "Stereocenters": len(chiral_centers)
            },
            "Drug_Likeness": {
                "HBA_Acceptors": hba,
                "HBD_Donors": hbd,
                "Lipinski_Rule_of_5_Violations": lipinski_violations,
                "Lipinski_Pass": lipinski_violations <= 1
            }
        }
        return analysis
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Node Analyzer & Descriptor Calculator")
    parser.add_argument("--name", type=str, help="Molecule name (IUPAC or common)")
    parser.add_argument("--smiles", type=str, help="SMILES string to analyze")
    args = parser.parse_args()

    if not args.name and not args.smiles:
        print(json.dumps({"error": "Please provide either --name or --smiles"}, indent=2))
        sys.exit(1)

    source = "LLM Direct SMILES"
    target_smiles = args.smiles

    if args.name and not target_smiles:
        target_smiles = get_smiles_from_pubchem(args.name)
        source = f"PubChem Resolution ('{args.name}')"
        if not target_smiles:
            print(json.dumps({"error": f"Could not resolve name '{args.name}' via PubChem API."}))
            sys.exit(1)

    result = analyze_smiles(target_smiles)
    if "error" not in result:
        result["Metadata"] = {"Source": source}
        
    # Output perfectly formatted JSON for Agent consumption
    print(json.dumps(result, indent=2))
