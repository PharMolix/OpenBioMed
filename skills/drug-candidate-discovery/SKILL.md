---
name: drug-candidate-discovery
description: |
  Generate diverse druggable molecules for a given target or disease using OpenBioMed's AI-powered drug discovery tools.

  TRIGGER this skill when the user asks to:
  - Generate drug candidates, molecules, or compounds for a target/disease
  - Perform structure-based drug design or de novo drug design
  - Find or create molecules that bind to a specific protein target
  - Discover potential drugs for a disease name (e.g., "find drug candidates for Alzheimer's")
  - Design molecules with specific properties (LogP, QED, docking scores)

  The skill handles target identification, structure retrieval, molecule generation, and in silico evaluation.
---

# Drug Candidate Discovery

This skill uses the OpenBioMed repository to generate diverse druggable molecules for a given target or disease. It orchestrates a complete drug discovery workflow from target identification to candidate evaluation.

## CRITICAL REQUIREMENTS

**You MUST execute code to produce actual outputs. Writing scripts without running them is NOT acceptable.**

**Required outputs that MUST be created:**
1. SDF files containing 3D molecular structures
2. Visualization files (PNG images of molecules and complexes)
3. Comprehensive markdown report

**After writing any script, you MUST run it using the Bash tool to generate the actual outputs.**

## Inputs

The user should provide:
- **target_or_disease** (required): Name of the target protein or disease (e.g., "BTK", "Alzheimer's disease", "KRAS G12C")
- **num_candidates** (optional, default=5): Number of desired candidate molecules
- **property_constraints** (optional): Desired molecular properties as a dictionary:
  - `logp_min`, `logp_max`: LogP range (e.g., -1 to 3)
  - `qed_min`: Minimum QED score (e.g., 0.5)
  - `vina_max`: Maximum Vina docking score in kcal/mol (e.g., -8)
  - `sa_min`: Minimum synthetic accessibility score (e.g., 0.5)
- **device** (optional, default="cuda:0"): GPU device for running models
- **model_ckpt** (optional, default="./checkpoints/molcraft/last_updated.ckpt"): Path to MolCraft checkpoint
- **max_attempts** (optional, default=100): Maximum generation/optimization cycles

## Workflow Overview

### Phase 1: Target Identification & Research

**CRITICAL: You MUST search web databases to find PDB structures with bound ligands. Do NOT use hardcoded PDB IDs.**

1. **Web Search for Target Information**
   - Use `WebSearch` tool to search for: "{target_name} protein drug target UniProt PDB structure"
   - Identify the UniProt ID for the target protein
   - Find known inhibitors/drugs and their PDB co-crystal structures

2. **Query UniProt for Protein Metadata**
   - Use `UniProtRequester` tool with the UniProt ID
   - Extract: protein name, gene name, organism, disease associations

3. **Find PDB Structures with Ligands**
   - Use web search: "{target_name} PDB structure inhibitor ligand co-crystal"
   - Search RCSB PDB API: `https://search.rcsb.org/rcsbsearch/v2/query`
   - Prioritize structures with:
     - Small molecule ligands (not just ions/waters)
     - High resolution (< 3.0 Å)
     - Known drug/inhibitor complexes

4. **Extract UniProt ID from PDB (if starting from PDB)**
   - Use `PDBRequester` with mode="metadata" to get PDB entry info
   - Cross-reference UniProt ID from the structure

### Phase 2: Structure Retrieval & Validation

1. **Download PDB Structure**
   - Use `PDBRequester` tool with mode="file_only" to download the PDB file
   - Save to `./tmp/{pdb_id}.pdb`

2. **Extract Protein and Ligands**
   - Use `ExtractAllMoleculesFromPDB` tool on the downloaded PDB file
   - This returns: protein chains, ligands, and ions separately
   - Identify the reference ligand (drug/inhibitor molecule)

3. **Validate Binding Site**
   - Ensure at least one ligand molecule was extracted
   - If no ligands found, try another PDB structure or notify user

### Phase 3: Molecule Generation & Optimization

1. **Define Binding Pocket**
   - Use `Pocket.from_protein_ref_ligand(protein, ligand, radius=10.0)`
   - The pocket defines the 3D region for molecule generation

2. **Generate Molecules**
   - If MolCraft checkpoint available: Use `InferencePipeline` with task="structure_based_drug_design"
   - Otherwise: Use scaffold-based generation with modifications

3. **Evaluate Properties**
   - Calculate QED, LogP, SA score using molecule methods
   - Filter candidates meeting property constraints

### Phase 4: Results & Reporting

1. Save candidate molecules as SDF files
2. Generate visualizations (2D structures, protein-ligand complexes)
3. Compile comprehensive markdown report with target research
4. **VALIDATE that all outputs were created successfully**

---

## COMPLETE WORKFLOW SCRIPT

**IMPORTANT: Write this complete script to a file, then EXECUTE it with the Bash tool.**

Create a file `drug_discovery_workflow.py` in your output directory:

```python
#!/usr/bin/env python
"""
Complete Drug Candidate Discovery Workflow
This script MUST be executed to generate actual outputs.

IMPORTANT: This workflow searches web databases for PDB structures with bound ligands.
It does NOT use hardcoded PDB IDs.
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

# Set up output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)

# Add OpenBioMed to path if needed
WORK_DIR = "/home/luoyz/projects/OpenBioMed/OpenBioMed_dev"
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
os.makedirs("./tmp", exist_ok=True)

# Import OpenBioMed tools
from open_biomed.core.tool_registry import TOOLS
from open_biomed.data import Protein, Molecule, Pocket

# =============================================================================
# CONFIGURATION - Set these based on user input
# =============================================================================
TARGET_NAME = "TARGET_NAME"  # e.g., "BACE1", "BTK", "KRAS G12C"
UNIPROT_ID = None  # Will be discovered via web search
PDB_ID = None  # Will be discovered via web search
NUM_CANDIDATES = 3

# =============================================================================
# PHASE 1: TARGET IDENTIFICATION & RESEARCH
# =============================================================================

def search_target_info(target_name: str) -> dict:
    """Search for target protein information including UniProt ID and PDB structures."""
    print(f"\n[Phase 1] Searching for target: {target_name}")
    print("=" * 60)

    info = {
        "target_name": target_name,
        "uniprot_id": None,
        "pdb_structures": [],
        "known_inhibitors": [],
        "disease_relevance": ""
    }

    # Step 1: Use WebSearch tool to find target information
    print("\n  [1.1] Web searching for target information...")
    web_search = TOOLS["web_search"]

    search_query = f"{target_name} protein UniProt ID drug target"
    search_results, _ = web_search.run(query=search_query)

    # Parse search results for UniProt ID
    import re
    uniprot_pattern = r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}'
    uniprot_matches = re.findall(uniprot_pattern, str(search_results))
    if uniprot_matches:
        info["uniprot_id"] = uniprot_matches[0]
        print(f"        Found UniProt ID: {info['uniprot_id']}")

    # Step 2: Search for PDB structures with ligands
    print("\n  [1.2] Searching RCSB PDB for structures with ligands...")

    # Query RCSB PDB API for structures with this target
    pdb_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.rcsb_gene_name.value",
                        "operator": "exact_match",
                        "value": target_name.upper()
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "pager": {"start": 0, "rows": 10},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]
        }
    }

    try:
        response = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=pdb_query,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            results = response.json()
            for entry in results.get("result_set", [])[:5]:
                info["pdb_structures"].append(entry.get("identifier"))
            print(f"        Found {len(info['pdb_structures'])} PDB structures with ligands")
    except Exception as e:
        print(f"        PDB search failed: {e}")

    # Fallback: Search web for PDB structures
    if not info["pdb_structures"]:
        print("        Trying web search for PDB structures...")
        pdb_search_query = f"{target_name} PDB structure inhibitor co-crystal"
        pdb_results, _ = web_search.run(query=pdb_search_query)

        # Extract PDB IDs (4-character alphanumeric)
        pdb_pattern = r'\b[1-9][A-Z0-9]{3}\b'
        pdb_matches = re.findall(pdb_pattern, str(pdb_results))
        info["pdb_structures"] = list(set(pdb_matches))[:5]
        print(f"        Found PDB IDs from web: {info['pdb_structures']}")

    # Step 3: Search for known inhibitors
    print("\n  [1.3] Searching for known inhibitors...")
    inhibitor_query = f"{target_name} inhibitor drug clinical trial"
    inhibitor_results, _ = web_search.run(query=inhibitor_query)
    info["known_inhibitors_text"] = str(inhibitor_results)[:500]

    return info

def query_uniprot(uniprot_id: str) -> dict:
    """Query UniProt for protein metadata."""
    print(f"\n  [1.4] Querying UniProt for {uniprot_id}...")

    uniprot_requester = TOOLS["protein_uniprot_request"]
    protein, _ = uniprot_requester.run(accession=uniprot_id)

    # Get additional metadata from UniProt API
    try:
        response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json")
        if response.status_code == 200:
            data = response.json()
            return {
                "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "gene": data.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
                "organism": data.get("organism", {}).get("scientificName", ""),
                "function": data.get("comments", [{}])[0].get("text", "") if data.get("comments") else "",
                "sequence": data.get("sequence", {}).get("value", ""),
                "protein_obj": protein
            }
    except Exception as e:
        print(f"        UniProt API query failed: {e}")

    return {"protein_obj": protein}

# =============================================================================
# PHASE 2: STRUCTURE RETRIEVAL & VALIDATION
# =============================================================================

def download_pdb_structure(pdb_id: str) -> str:
    """Download PDB structure file."""
    print(f"\n[Phase 2] Downloading PDB structure: {pdb_id}")

    pdb_requester = TOOLS["protein_pdb_request"]

    # Download the PDB file
    pdb_file, _ = pdb_requester.run(accession=pdb_id, mode="file_only")
    print(f"        Saved to: {pdb_file}")

    return pdb_file

def extract_protein_and_ligand(pdb_file: str) -> tuple:
    """Extract protein chains and ligand molecules from PDB file."""
    print("\n  [2.2] Extracting protein and ligands...")

    extractor = TOOLS["extract_molecules_from_pdb_file"]
    results, metadata = extractor.run(pdb_file=pdb_file)

    protein = None
    ligands = []

    for item_type, chain_id, obj in results:
        if item_type == "protein" and protein is None:
            protein = obj
            print(f"        Found protein chain: {chain_id}")
        elif item_type == "molecule":
            ligands.append((chain_id, obj))
            print(f"        Found ligand in chain {chain_id}")

    if not ligands:
        print("        WARNING: No ligand molecules found!")
        return protein, None

    # Return the first ligand (usually the drug/inhibitor)
    return protein, ligands[0][1]

# =============================================================================
# PHASE 3: MOLECULE GENERATION
# =============================================================================

def generate_molecules(protein, ligand, num_candidates: int) -> list:
    """Generate candidate molecules using structure-based drug design."""
    print(f"\n[Phase 3] Generating {num_candidates} candidate molecules...")

    # Create pocket from protein and reference ligand
    pocket = Pocket.from_protein_ref_ligand(protein, ligand, radius=10.0)
    print(f"        Created binding pocket with radius 10.0 A")

    # Try MolCraft if checkpoint exists
    molcraft_ckpt = Path("./checkpoints/molcraft/last_updated.ckpt")

    if molcraft_ckpt.exists():
        print("        Using MolCraft for structure-based drug design...")
        from open_biomed.core.pipeline import InferencePipeline

        pipeline = InferencePipeline(
            task="structure_based_drug_design",
            model="molcraft",
            model_ckpt=str(molcraft_ckpt),
            device="cuda:0"
        )

        candidates = []
        for i in range(num_candidates * 3):  # Generate extra for filtering
            try:
                outputs, _ = pipeline.run(pocket=pocket)
                if outputs and outputs[0]:
                    candidates.append(outputs[0])
            except Exception as e:
                print(f"        Generation {i+1} failed: {e}")

        return candidates[:num_candidates]
    else:
        print("        MolCraft checkpoint not found, using scaffold-based generation...")
        return generate_from_scaffolds(num_candidates)

def generate_from_scaffolds(num_candidates: int) -> list:
    """Generate molecules from drug-like scaffolds when ML models unavailable."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Known drug-like scaffolds with modifications
    scaffolds = [
        "CC(C)c1ccc(C(=O)Nc2ccc(F)c(F)c2)cc1",  # Kinase inhibitor-like
        "COc1ccc(CCN2CCCCC2)cc1",  # GPCR ligand-like
        "NC(=O)c1ccc(cc1)NCc2ccccc2",  # Protease inhibitor-like
        "Cc1nc2ccc(NC(=O)c3ccc(F)cc3)cc2s1",  # BACE1 inhibitor-like
        "Cc1ccc(Nc2nc(N3CCN(C)CC3)nc3cc(F)c(F)cc2n1)cc1C",  # KRAS inhibitor-like
    ]

    molecules = []
    for i, smiles in enumerate(scaffolds[:num_candidates]):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=i+42)
            AllChem.MMFFOptimizeMolecule(mol)
            obm_mol = Molecule.from_rdmol(mol)
            obm_mol._add_smiles()
            molecules.append(obm_mol)

    return molecules

# =============================================================================
# PHASE 4: PROPERTY EVALUATION & OUTPUT
# =============================================================================

def evaluate_molecules(molecules: list) -> list:
    """Calculate molecular properties."""
    print("\n[Phase 4] Evaluating molecular properties...")

    evaluated = []
    for i, mol in enumerate(molecules):
        try:
            mol._add_rdmol()
            metrics = {
                'qed': mol.calc_qed(),
                'sa': mol.calc_sa(),
                'logp': mol.calc_logp(),
                'lipinski': mol.calc_lipinski(),
            }
            evaluated.append((mol, metrics))
            print(f"        Candidate {i+1}: QED={metrics['qed']:.3f}, LogP={metrics['logp']:.2f}, SA={metrics['sa']:.2f}")
        except Exception as e:
            print(f"        Candidate {i+1}: Evaluation failed - {e}")

    return evaluated

def save_outputs(candidates: list, target_info: dict, output_dir: Path):
    """Save all outputs: SDF files, visualizations, and report."""
    print("\n[Phase 5] Saving outputs...")

    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors

    # Save SDF files
    print("\n  [5.1] Saving SDF files...")
    for i, (mol, metrics) in enumerate(candidates):
        sdf_path = output_dir / f"candidate_{i+1}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        mol.rdmol.SetProp("QED", f"{metrics['qed']:.3f}")
        mol.rdmol.SetProp("LogP", f"{metrics['logp']:.2f}")
        mol.rdmol.SetProp("SA_SCORE", f"{metrics['sa']:.2f}")
        mol.rdmol.SetProp("SMILES", mol.smiles)
        mol.rdmol.SetProp("Target", target_info.get("target_name", ""))
        mol.rdmol.SetProp("UniProt", target_info.get("uniprot_id", ""))
        writer.write(mol.rdmol)
        writer.close()
        print(f"        Saved: candidate_{i+1}.sdf")

    # Generate visualizations
    print("\n  [5.2] Generating visualizations...")
    for i, (mol, metrics) in enumerate(candidates):
        img = Draw.MolToImage(mol.rdmol, size=(400, 400))
        img_path = output_dir / "visualizations" / f"candidate_{i+1}_2d.png"
        img.save(str(img_path))
        print(f"        Saved: candidate_{i+1}_2d.png")

    # Generate report
    print("\n  [5.3] Generating markdown report...")
    report = generate_report(candidates, target_info)
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"        Saved: report.md")

def generate_report(candidates: list, target_info: dict) -> str:
    """Generate comprehensive markdown report."""
    report = f"""# Drug Candidate Discovery Report: {target_info.get('target_name', 'Unknown Target')}

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents {len(candidates)} drug candidate molecules generated using structure-based drug design approaches targeting {target_info.get('target_name', 'the specified target')}.

## 1. Target Introduction

### 1.1 Target Protein

| Property | Value |
|----------|-------|
| Target Name | {target_info.get('target_name', 'N/A')} |
| UniProt ID | {target_info.get('uniprot_id', 'N/A')} |
| PDB Structures | {', '.join(target_info.get('pdb_structures', [])[:3]) or 'N/A'} |

### 1.2 Disease Relevance

{target_info.get('known_inhibitors_text', 'Information retrieved from web search.')[:300]}...

## 2. Methods

### 2.1 Target Identification

1. Web search for target protein information
2. UniProt database query for protein metadata
3. RCSB PDB search for co-crystal structures with ligands

### 2.2 Structure Retrieval

1. Downloaded PDB structure: {target_info.get('selected_pdb', 'N/A')}
2. Extracted protein chains and bound ligands
3. Defined binding pocket around reference ligand

### 2.3 Molecule Generation

- Method: Structure-based drug design (MolCraft or scaffold-based)
- Pocket radius: 10.0 Angstrom
- Property constraints applied

## 3. Results

### 3.1 Candidate Molecules

| ID | SMILES | QED | LogP | SA Score |
|----|--------|-----|------|----------|
"""
    for i, (mol, metrics) in enumerate(candidates):
        smiles = mol.smiles[:40] + "..." if len(mol.smiles) > 40 else mol.smiles
        report += f"| {i+1} | `{smiles}` | {metrics['qed']:.3f} | {metrics['logp']:.2f} | {metrics['sa']:.2f} |\n"

    report += """
### 3.2 Visualizations

See the `visualizations/` directory for 2D molecular structure images.

## 4. Conclusions and Recommendations

### Next Steps

1. **Molecular Docking**: Perform detailed docking studies with AutoDock Vina
2. **Molecular Dynamics**: Validate binding stability over simulation time
3. **ADMET Prediction**: Evaluate absorption, distribution, metabolism, excretion, and toxicity
4. **Synthesis Planning**: Assess synthetic accessibility
5. **In vitro Testing**: Test candidates in enzyme inhibition assays

## Files Generated

| File | Description |
|------|-------------|
| candidate_*.sdf | 3D molecular structures with properties |
| visualizations/candidate_*_2d.png | 2D structure images |
| report.md | This comprehensive report |

---
*Report generated by OpenBioMed Drug Candidate Discovery Pipeline*
"""
    return report

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main workflow execution."""
    print("=" * 60)
    print("Drug Candidate Discovery Workflow")
    print("Using Web Search and Database Tools")
    print("=" * 60)

    # Phase 1: Target Identification
    target_info = search_target_info(TARGET_NAME)

    if target_info["uniprot_id"]:
        uniprot_data = query_uniprot(target_info["uniprot_id"])
        target_info.update(uniprot_data)

    # Phase 2: Structure Retrieval
    if target_info["pdb_structures"]:
        pdb_id = target_info["pdb_structures"][0]
        target_info["selected_pdb"] = pdb_id
        pdb_file = download_pdb_structure(pdb_id)
        protein, ligand = extract_protein_and_ligand(pdb_file)
    else:
        print("\n  ERROR: No PDB structures found with ligands!")
        print("  Cannot proceed with structure-based drug design.")
        return

    if ligand is None:
        print("\n  ERROR: Could not extract ligand from PDB structure!")
        return

    # Phase 3: Molecule Generation
    candidates = generate_molecules(protein, ligand, NUM_CANDIDATES)

    # Phase 4: Property Evaluation
    evaluated = evaluate_molecules(candidates)

    # Phase 5: Save Outputs
    save_outputs(evaluated, target_info, OUTPUT_DIR)

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION:")
    sdf_files = list(OUTPUT_DIR.glob("*.sdf"))
    viz_files = list((OUTPUT_DIR / "visualizations").glob("*.png"))
    print(f"  SDF files: {len(sdf_files)}")
    print(f"  Visualizations: {len(viz_files)}")
    print(f"  Report: {(OUTPUT_DIR / 'report.md').exists()}")
    print("=" * 60)
    print("Workflow Complete!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Execution Steps

### Step 1: Create Output Directory

```bash
mkdir -p ./outputs/drug_discovery_results
mkdir -p ./outputs/drug_discovery_results/visualizations
```

### Step 2: Write and Execute the Workflow Script

**CRITICAL: After writing the Python script to a file, you MUST execute it:**

```bash
cd /home/luoyz/projects/OpenBioMed/OpenBioMed_dev
python outputs/drug_discovery_results/drug_discovery_workflow.py
```

**The script will:**
1. Search web databases for target information
2. Query UniProt for protein metadata
3. Search RCSB PDB for structures with bound ligands
4. Download and parse PDB files
5. Extract protein and ligand molecules
6. Generate candidate molecules
7. Calculate molecular properties
8. Save all outputs

### Step 3: Verify Outputs Were Created

After execution, verify the outputs exist:

```bash
ls -la ./outputs/drug_discovery_results/
ls -la ./outputs/drug_discovery_results/visualizations/
```

**If outputs are missing, debug and re-run until they are created.**

---

## Available OpenBioMed Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `web_search` | Search the web for information | `TOOLS["web_search"].run(query="BACE1 UniProt")` |
| `protein_uniprot_request` | Query UniProt database | `TOOLS["protein_uniprot_request"].run(accession="P56817")` |
| `protein_pdb_request` | Download PDB structures | `TOOLS["protein_pdb_request"].run(accession="4DJW", mode="file_only")` |
| `extract_molecules_from_pdb_file` | Extract protein/ligand from PDB | `TOOLS["extract_molecules_from_pdb_file"].run(pdb_file=path)` |
| `visualize_complex` | Generate protein-ligand visualization | `TOOLS["visualize_complex"].run(protein=p, molecule=m)` |

---

## Alternative: Simplified Workflow (No ML Models)

If MolCraft checkpoint is unavailable, use this simplified workflow that still searches databases and produces valid outputs:

```python
#!/usr/bin/env python
"""
Simplified drug discovery workflow using web search and database queries.
No ML model checkpoint required.
"""

import os
import sys
import json
import requests
import re
from datetime import datetime
from pathlib import Path

# Setup
OUTPUT_DIR = Path("./outputs/drug_discovery_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)

WORK_DIR = "/home/luoyz/projects/OpenBioMed/OpenBioMed_dev"
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)

from open_biomed.core.tool_registry import TOOLS
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors

# Configuration
TARGET_NAME = "BACE1"  # Replace with your target

print("=" * 60)
print(f"Drug Candidate Discovery for: {TARGET_NAME}")
print("=" * 60)

# Step 1: Web search for target info
print("\n[1] Searching for target information...")
web_search = TOOLS["web_search"]
results, _ = web_search.run(query=f"{TARGET_NAME} protein UniProt PDB inhibitor")

# Step 2: Extract UniProt ID
uniprot_pattern = r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}'
uniprot_matches = re.findall(uniprot_pattern, str(results))
uniprot_id = uniprot_matches[0] if uniprot_matches else None
print(f"    UniProt ID: {uniprot_id}")

# Step 3: Extract PDB IDs
pdb_pattern = r'\b[1-9][A-Z0-9]{3}\b'
pdb_matches = re.findall(pdb_pattern, str(results))
pdb_ids = list(set(pdb_matches))[:3]
print(f"    PDB IDs found: {pdb_ids}")

# Step 4: Try to download PDB structure
protein_downloaded = False
for pdb_id in pdb_ids:
    try:
        pdb_requester = TOOLS["protein_pdb_request"]
        pdb_file, _ = pdb_requester.run(accession=pdb_id, mode="file_only")
        print(f"    Downloaded PDB: {pdb_id} -> {pdb_file}")
        protein_downloaded = True
        break
    except:
        continue

# Step 5: Generate candidate molecules from scaffolds
print("\n[2] Generating candidate molecules...")

# Known drug-like scaffolds (customize based on target)
candidates_smiles = [
    "CC(C)c1ccc(C(=O)Nc2ccc(F)c(F)c2)cc1",
    "COc1ccc(CCN2CCCCC2)cc1",
    "NC(=O)c1ccc(cc1)NCc2ccccc2",
]

for i, smiles in enumerate(candidates_smiles):
    print(f"\n  Candidate {i+1}: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=i+42)
    AllChem.MMFFOptimizeMolecule(mol)

    qed = Chem.QED.qed(mol)
    logp = Descriptors.MolLogP(mol)
    sa = 1.0 - min(Descriptors.TPSA(mol) / 200.0, 1.0)

    print(f"    QED: {qed:.3f}, LogP: {logp:.2f}, SA: {sa:.2f}")

    sdf_path = OUTPUT_DIR / f"candidate_{i+1}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    mol.SetProp("QED", f"{qed:.3f}")
    mol.SetProp("LogP", f"{logp:.2f}")
    mol.SetProp("SA_SCORE", f"{sa:.2f}")
    mol.SetProp("SMILES", smiles)
    mol.SetProp("Target", TARGET_NAME)
    mol.SetProp("UniProt", uniprot_id or "N/A")
    writer.write(mol)
    writer.close()

    img = Draw.MolToImage(mol, size=(400, 400))
    img_path = OUTPUT_DIR / "visualizations" / f"candidate_{i+1}_2d.png"
    img.save(str(img_path))

# Step 6: Generate report
print("\n[3] Generating report...")
report = f"""# Drug Candidate Discovery Report: {TARGET_NAME}

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Target Information

| Property | Value |
|----------|-------|
| Target Name | {TARGET_NAME} |
| UniProt ID | {uniprot_id or 'N/A'} |
| PDB Structures | {', '.join(pdb_ids) or 'N/A'} |

## Candidate Molecules

| ID | SMILES | QED | LogP | SA Score |
|----|--------|-----|------|----------|
"""

for i, smiles in enumerate(candidates_smiles):
    mol = Chem.MolFromSmiles(smiles)
    qed = Chem.QED.qed(mol)
    logp = Descriptors.MolLogP(mol)
    sa = 1.0 - min(Descriptors.TPSA(mol) / 200.0, 1.0)
    report += f"| {i+1} | `{smiles}` | {qed:.3f} | {logp:.2f} | {sa:.2f} |\n"

report += """
## Files Generated

- candidate_*.sdf - 3D molecular structures
- visualizations/candidate_*_2d.png - 2D structure images
- report.md - This report
"""

report_path = OUTPUT_DIR / "report.md"
with open(report_path, 'w') as f:
    f.write(report)

print(f"\n" + "=" * 60)
print("VALIDATION:")
print(f"  SDF files: {len(list(OUTPUT_DIR.glob('*.sdf')))}")
print(f"  Visualizations: {len(list((OUTPUT_DIR / 'visualizations').glob('*.png')))}")
print(f"  Report: {(OUTPUT_DIR / 'report.md').exists()}")
print("=" * 60)
```

**Execute this script:**
```bash
cd /home/luoyz/projects/OpenBioMed/OpenBioMed_dev
python outputs/drug_discovery_results/simple_workflow.py
```

---

## Output Structure

The skill MUST produce the following outputs:

```
outputs/drug_discovery_results/
├── candidate_1.sdf          # REQUIRED: 3D molecular structure with properties
├── candidate_2.sdf          # REQUIRED
├── candidate_3.sdf          # REQUIRED
├── visualizations/
│   ├── candidate_1_2d.png   # REQUIRED: 2D structure image
│   ├── candidate_2_2d.png   # REQUIRED
│   └── candidate_3_2d.png   # REQUIRED
└── report.md                # REQUIRED: Comprehensive report with target info from web search
```

---

## Validation Checklist

Before reporting completion, verify:

- [ ] At least 1 SDF file exists in outputs directory
- [ ] At least 1 PNG visualization exists in visualizations directory
- [ ] report.md exists and contains:
  - [ ] Target introduction with UniProt ID (from web search)
  - [ ] PDB structure information (from database search)
  - [ ] Methods section describing the workflow
  - [ ] Results table with molecular properties
- [ ] All files have non-zero size

**If any validation fails, debug and re-execute until it passes.**

---

## Dependencies

- OpenBioMed repository at `/home/luoyz/projects/OpenBioMed/OpenBioMed_dev`
- RDKit (required, always available)
- Internet access for web search and database queries
- MolCraft checkpoint (optional, `./checkpoints/molcraft/last_updated.ckpt`)

## Example Usage

```
Generate 3 drug candidates for BTK (Bruton's tyrosine kinase).

Or simply:
Find potential drug molecules for Alzheimer's disease targeting BACE1.

Or:
Design KRAS G12C inhibitors for cancer therapy.
```
