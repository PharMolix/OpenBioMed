---
name: functional-protein-design
name: functional-protein-design
description: >
  Generate functional protein sequences. Use this skill when:
  (1) Generating de novo protein sequences guided by specific Gene Ontology (GO) tags.
  (2) Exploring sequence space with prior functional constraints.
license: MIT
category: design-tools
tags: [protein-design, go-guided, sequence-generation, structure-prediction]
---

# Functional Protein Design

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Environment | Configured via OpenBioMed: [`README.md`](https://github.com/PharMolix/OpenBioMed/blob/main/README.md) |
| Hardware | CUDA-compatible GPU (≥ 10GB VRAM) required for both generation and folding |
| Checkpoints | Download CodeFP weights & mappings from [Google Drive](https://drive.google.com/drive/folders/1Zqp2uD-f3cSzXeg35ixK-Epf-HpKBQYY?usp=sharing). |

## Data Preparation & Configuration

**Directory Structure**
Organize your downloaded checkpoints and mapping files as follows:

```text
checkpoints/
├── codefp/
│   ├── model/
│   │   └── checkpoints/
│   │       └── model.ckpt
│   └── mappings/
│       ├── go_mapping.pkl
│       ├── go_id_mapping.pkl
│       ├── desc2map_dict_statics.pkl
│       └── train_go_terms_cls_emb.pkl

```

## How to Run

### Phase 0: Environment & Prerequisites

Before getting started, ensure that your environment is fully configured. This includes a successful installation of OpenBioMed and the completion of all required model weight downloads.

Next, search the [Gene Ontology website](https://geneontology.org/docs/ontology-documentation/) to identify 1–3 Molecular Function (MF) GO terms (e.g., ['GO:0004930', 'GO:0004984']) that best align with your functional target.

Note: Please ensure that the selected GO terms are included in go_mapping.pkl, a dictionary whose keys enumerate all supported GO terms (e.g., “GO:0004930”, “GO:0004984”), to ensure compatibility with the model.

### Phase 1: GO-Guided Sequence Generation (Python)

First, we generate the protein sequence using the model. Run the following code:

```python
from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Protein

# 1. GO-guided sequence generation
generator = InferencePipeline(
    task="go_guided_protein_generation",
    model="codefp",
    model_ckpt="./checkpoints/codefp/model/checkpoints/model.ckpt",
    device="cuda:0"
)

# Replace with 1–3 target Molecular Function (MF) GO terms
go_terms = [['GO:0004930', 'GO:0004984']]

designed_seqs = generator.run(go_terms=go_terms)
seq_only = designed_seqs[0][0]  # Protein object

seq_str = seq_only.sequence
print(f"Generated Sequence: {seq_str}")
```

### Phase 2: Functional Evaluation & Quality Check via InterProScan (Terminal Execution)

Critical Checkpoint: Before proceeding to structure prediction, you must use InterProScan to verify whether the generated sequence (seq_str) possesses the expected function.

Steps:

1. Download the InterProScan client script: 

```bash
curl -L -o iprscan5.py https://raw.githubusercontent.com/ebi-jdispatcher/webservice-clients/master/python/iprscan5.py
```
2. Run the following command in your Terminal (replace your@email.com and paste the actual sequence generated in Phase 1):

```bash
python iprscan5.py --email your@email.com \
  --stype p \
  --sequence "PASTE_THE_SEQ_STR_GENERATED_IN_PHASE_1_HERE" \
  --outformat json
```

Workflow Decision Tree:

Check the generated JSON annotation file:

    ✅ Target GO terms appear: Validation successful! Proceed to Phase 3.

    ❌ Target GO terms DO NOT appear: Validation failed. Return to Phase 1 to regenerate sequences.

    🔄 If unsuccessful after 3 attempts: Select the generated sequence whose functional annotations most closely match your target function, then proceed.

### Phase 3: Structure Prediction & pLDDT Writing (Python)

Once validated, input the sequence into ESMFold for 3D structure prediction, and write the pLDDT confidence scores into the B-factor field of the resulting PDB file.

```python

folder = InferencePipeline(
    task="protein_folding",
    model="esmfold",
    model_ckpt="./checkpoints/server/esmfold.ckpt", # ESMFold will be downloaded automatically.
    device="cuda:0"
)

seq_only = Protein.from_fasta(seq_str)

folded_protein = folder.run(protein=seq_only)
plddt = folder.model.model.temp_plddt_res[0]
mean_plddt = plddt.mean()
print(f"Global pLDDT: {mean_plddt:.2f}")


folded_protein[0][0].save_pdb("designed.pdb", overwrite=True)


from Bio import PDB

def update_plddt_with_biopython(input_pdb, output_pdb, plddt_array):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    
    print(f"Global pLDDT: {plddt_array.mean():.4f}")
    
    residues = list(structure.get_residues())
    
    if len(residues) != len(plddt_array):
        print(f"Warning: PDB residue count ({len(residues)}) does not match pLDDT array length ({len(plddt_array)}).")

    for i, res in enumerate(residues):
        if i < len(plddt_array):
            score = plddt_array[i]  
            for atom in res:
                atom.set_bfactor(score)
                
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Successfully saved updated PDB to: {output_pdb}")

update_plddt_with_biopython("designed.pdb", "designed.pdb", plddt * 100)

```

### Phase 4: (Optional) PyMOL Structure Visualization (Python)

```python
import os
import shutil
from datetime import datetime
from pymol import cmd

def visualize_protein_plddt(pdb_file, output_gif, num_frames=36):
    try:
        cmd.reinitialize()
        
       
        cmd.load(pdb_file, "prot_obj")
        
        
        cmd.hide("everything", "prot_obj") 
        cmd.show_as("cartoon", "prot_obj")
        
        
        cmd.set_color("db_blue",   [0.00, 0.33, 0.71])
        cmd.set_color("db_cyan",   [0.39, 0.74, 0.93])
        cmd.set_color("db_yellow", [1.00, 0.78, 0.16])
        cmd.set_color("db_orange", [1.00, 0.48, 0.30])
        
        cmd.color("db_blue",   "prot_obj and b > 90")
        cmd.color("db_cyan",   "prot_obj and b < 90")
        cmd.color("db_yellow", "prot_obj and b < 70")
        cmd.color("db_orange", "prot_obj and b < 50")

   
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.orient("prot_obj") 
        cmd.zoom("prot_obj", buffer=2.0)

       
        temp_dir = f"temp_pngs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Generating frames in {temp_dir}...")


        for i in range(num_frames):

            cmd.rotate("y", 360.0 / num_frames, "prot_obj")
            
            frame_path = os.path.join(temp_dir, f"frame{i:03d}.png")

            cmd.png(frame_path, width=800, height=600, dpi=150, ray=0)


        from open_biomed.tools.visualization_tools import convert_png2gif
        
        if len(os.listdir(temp_dir)) > 0:
            convert_png2gif(temp_dir, output_gif, fps=12)
            print(f"Success! Saved to {output_gif}")
        else:
            print("Error: No PNG frames were generated.")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    except ImportError as ie:
        print(f"Import Error: Please make sure PyMOL and open_biomed are installed. Details: {ie}")
    except Exception as e:
        print(f"An error occurred: {e}")
visualize_protein_plddt("designed.pdb", "designed_rotated.gif")

```

## Expected Deliverables

Every successful run must yield a report containing:

1. **The generated PDB file** (`designed.pdb`).
3. **The PNG image** (`designed.png`) rendered from the PDB file.
4. **The GIF image** (`designed_rotated.gif`) rendered from the PDB file.
2. **The exact GO tags used** for generation.
3. **Brief descriptions** of each GO tag.

**Sample Deliverable Report:**

* **Used GO Terms:** `['GO:0004930', 'GO:0004984']`
* **Descriptions:** * `GO:0004930` — G protein-coupled receptor activity
* `GO:0004984` — Olfactory receptor activity



## Troubleshooting

| Error / Warning | Cause | Fix |
| --- | --- | --- |
| `Warning: "GO ID {go_id} not found in mapping, using hash instead."` | Target GO ID is not supported in `go_mapping.pkl`. | **1.** Find the closest alternative GO combination in `go_mapping.pkl`.<br>**2.** Rerun with the alternative.<br>**3.** Explicitly report the substituted GO combination to the user. |
| `FileNotFoundError` or Checkpoint fails to load | Incorrect paths in `codefp.yaml`. | Verify file paths match the actual directory structure. |
