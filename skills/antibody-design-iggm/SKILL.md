---
name: antibody-design-iggm
description: >
  Antibody design using IgGM model.
  Use this skill when:
  (1) Epitope-conditioned de novo antibody design,
  (2) Antibody affinity maturation,
  (3) Using antigen PDB structure and epitope information.

  For binding affinity evaluation, use prodigy.
license: MIT
category: design-tools
tags: [structure-design, sequence-design, antibody, nanobody]
---

# IgGM Antibody De Novo Design

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.10 |
| CUDA | 11.7+ | 11.8 |
| GPU VRAM | 24GB | 80GB (A800) |
| RAM | 32GB | 64GB |

## How to run

### Local installation
```bash
git clone https://github.com/TencentAI4S/IgGM.git
cd IgGM

pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install tqdm requests numpy==1.23.5 termcolor==2.4.0 biopython==1.79 openmm==8.2 pdbfixer ml-collections==0.1.1
```

### Predict epitope based on antigen-antibody complex structure
```bash
python design.py --fasta complex_sequence.fasta --antigen complex_structure.pdb --cal_epitope
# --antigen: the structure of a known complex
# --fasta: the sequence of a known complex
```
* The generated epitope format is (The serial number starts at 1): 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 
* If you specify epitope according to the sequence, make sure that the order of the sequence is consistent with the order in the PDB file, and mark the serial number of the corresponding position.

### Given the structure of an antigen, design an antibody
```bash
python design.py --fasta design_requirement.fasta --antigen antigen_structure.pdb --epitope 7 8 9 10 11 --output output_dir
# --fasta: Directory path to input design requirement FASTA files, X for design region
# --antigen: Directory path to input antigen PDB files
# --epitope: epitope residues in antigen chain A , for example: 7 8 9 10 11
# --output: Directory path to output PDB files
```

### Affinity maturation for an antibody sequence
```bash
python design.py --fasta design_requirement.fasta --antigen antigen_structure.pdb --fasta_origin original_antibody_sequence.fasta --run_task affinity_maturation --num_samples 10 --output output_dir
# --fasta: Directory path to input design requirement FASTA files, X for design region
# --antigen: Directory path to input antigen PDB files
# --fasta_origin: Directory path to original antibody FASTA files for affinity maturation
# --num_samples: number of samples for residue
# --output: Directory path to output PDB files
```

## FASTA format
```
>H  # Heavy chain ID
VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS
>L # Light chain ID
DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK
>A # Antigen ID, needs to be consistent with the pdb file
NLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGP
```
* 'X' indicates the region to be designed.

## Output format

```
output_dir/
├── antibody_0.fasta # designed antibody sequence (fasta file)
├── antibody_0.pdb   # designed antigen-antibody complex structure (pdb file)
├── antibody_1.fasta 
├── antibody_1.pdb   
└── ...
```

## Decision tree

```
Should I use IgGM?
│
└─ What type of design?
   ├─ Antibody de novo design → IgGM ✓
   ├─ Nanobody de novo design → IgGM ✓
   └─ Binder design for general protein binder → boltzgen
```

**Next**: Evaluate binding affinity with `prodigy`.