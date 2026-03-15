---
name: boltzgen
description: >
  All-atom protein design using BoltzGen diffusion model.
  Use this skill when:
  (1) Need side-chain aware design from the start,
  (2) Designing around small molecules or ligands,
  (3) Want all-atom diffusion (not just backbone),
  (4) Require precise binding geometries,
  (5) Using YAML-based configuration.

  For structure validation, use boltz-2.
license: MIT
category: design-tools
tags: [structure-design, sequence-design, diffusion, all-atom, binder]
---

# BoltzGen All-Atom Design

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12 |
| CUDA | 12.0+ | 12.2 |
| GPU VRAM | 24GB | 80GB (A800) |
| RAM | 32GB | 64GB |

## How to run

### Local installation
```bash
git clone https://github.com/HannesStark/boltzgen.git
cd boltzgen
pip install -e .
```

### Binder design according to the input YAML file
```bash
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2
# --num_designs is the number of intermediate designs. In practice you will want between 10,000 - 60,000
# --budget is how many designs should be in the final diversity optimized set
```

## YAML configuration

BoltzGen uses an entity-based YAML format to specify what to design and what the target is.

**Important notes:**
- Residue indices use `label_seq_id` (1-indexed), not `auth_seq_id`
- File paths are relative to the YAML file location
- Run `boltzgen check config.yaml` to verify before running
- View in Molstar to confirm binding site is correctly specified

## Entity Types

### Designed Protein
```yaml
entities:
  - protein:
      id: B                    # Chain ID for designed protein
      sequence: 80..140        # Variable length (80-140 residues)
```

**Sequence specification:**
- `80..140` - random length between 80 and 140 residues
- `80` - exactly 80 designed residues
- `AAAVVV20PPP` - specific residues with 20 designed in middle
- `3..5C6C3` - designed residues with specific cysteines

### Target from File
```yaml
entities:
  - file:
      path: target.cif        # CIF or PDB file (relative to YAML)
      include:                 # Which chains/residues to include
        - chain:
            id: A
            res_index: 2..50,55..  # Optional: specific residues
      binding_types:           # Where design should bind
        - chain:
            id: A
            binding: 45,67,89  # Binding site residues
      structure_groups: "all"  # Optional: structure specification
```

### Non-Designed Protein
```yaml
entities:
  - protein:
      id: X
      sequence: AAVTTTTPPP    # Fixed sequence (not designed)
```

### Constraints (Bonds)
```yaml
constraints:
  - bond:
      atom1: [S, 11, SG]      # [chain_id, res_index, atom_name]
      atom2: [S, 18, SG]      # Disulfide bond
```

## Protocol-Specific Examples

### Protein Binder Design (`protein-anything`)
```yaml
entities:
  # Designed binder (80-140 residues)
  - protein:
      id: B
      sequence: 80..140

  # Target protein
  - file:
      path: target.cif
      include:
        - chain:
            id: A
      binding_types:
        - chain:
            id: A
            binding: 45,67,89
```

### Peptide Design (`peptide-anything`)
```yaml
entities:
  # Designed peptide (12-20 residues)
  - protein:
      id: G
      sequence: 12..20

  - file:
      path: target.cif
      include:
        - chain:
            id: A
      binding_types:
        - chain:
            id: A
            binding: 343,344,251
      structure_groups: "all"
```

### Cyclic Peptide with Disulfide
```yaml
entities:
  - protein:
      id: S
      sequence: 10..14C6C3    # Designed with cysteines

  - file:
      path: target.cif
      include:
        - chain:
            id: A

constraints:
  - bond:
      atom1: [S, 11, SG]
      atom2: [S, 18, SG]
```

### WHL Stapled Peptide
```yaml
entities:
  - protein:
      id: R
      sequence: 3..5C6C3

  - ligand:
      id: Q
      ccd: WHL

  - file:
      path: target.cif
      include:
        - chain:
            id: A

constraints:
  - bond:
      atom1: [R, 4, SG]
      atom2: [Q, 1, CK]
  - bond:
      atom1: [R, 11, SG]
      atom2: [Q, 1, CH]
```

### Small Molecule Binding (`protein-small_molecule`)
```yaml
entities:
  - protein:
      id: A
      sequence: 100..150

  - ligand:
      smiles: "CCO"           # Ethanol
      # or ccd: ATP           # From CCD database
```

### Nanobody Design (`nanobody-anything`)
```yaml
entities:
  - protein:
      id: H
      sequence: EVQLVESGG...  # Framework with designed CDRs
      # Use specific residue notation for CDR design

  - file:
      path: antigen.cif
      include:
        - chain:
            id: A
```

## Advanced Options

### Partial Target Flexibility
```yaml
entities:
  - file:
      path: target.cif
      include:
        - chain:
            id: A
      structure_groups:
        - group:
            visibility: 1     # Fixed structure
            id: A
            res_index: 10..50
        - group:
            visibility: 0     # Flexible (not structurally specified)
            id: A
            res_index: 51..60
```

### Redesign Existing Residues
```yaml
entities:
  - file:
      path: complex.cif
      include:
        - chain:
            id: A
      design:                  # Residues to redesign
        - chain:
            id: A
            res_index: 14..19
```

### Secondary Structure Constraints
```yaml
entities:
  - file:
      path: target.cif
      design:
        - chain:
            id: A
            res_index: 14..19
      secondary_structure:
        - chain:
            id: A
            helix: 15..17
            sheet: 19
            loop: 14
```

### Not-Binding Regions
```yaml
entities:
  - file:
      path: target.cif
      include:
        - chain:
            id: A
        - chain:
            id: B
      binding_types:
        - chain:
            id: A
            binding: 45,67,89
        - chain:
            id: B
            not_binding: "all"  # Design should NOT bind here
```

## Design protocols

| Protocol | Use Case |
|----------|----------|
| `protein-anything` | Design proteins to bind proteins or peptides |
| `peptide-anything` | Design cyclic peptides to bind proteins |
| `protein-small_molecule` | Design proteins to bind small molecules |
| `nanobody-anything` | Design nanobody CDRs |
| `antibody-anything` | Design antibody CDRs |

## Output format

```
output/
├── sample_0/
│   ├── design.cif         # All-atom structure (CIF format)
│   ├── metrics.json       # Confidence scores
│   └── sequence.fasta     # Sequence
├── sample_1/
│   └── ...
└── summary.csv
```

**Note**: BoltzGen outputs CIF format. Convert to PDB if needed:
```python
from Bio.PDB import MMCIFParser, PDBIO
parser = MMCIFParser()
structure = parser.get_structure("design", "design.cif")
io = PDBIO()
io.set_structure(structure)
io.save("design.pdb")
```

## Sample output

### Successful run
```
$ modal run modal_boltzgen.py --input-yaml binder.yaml --protocol protein-anything --num-designs 10
Running: boltzgen run binder.yaml --output /tmp/out --protocol protein-anything --num_designs 10
[INFO] Loading BoltzGen model...
[INFO] Generating designs...
[INFO] Running inverse folding...
[INFO] Running structure prediction...
[INFO] Filtering and ranking...
[INFO] Pipeline complete

Results saved to: ./out/boltzgen/2501161234/
```

**Output directory structure:**
```
out/boltzgen/2501161234/
├── intermediate_designs/           # Raw diffusion outputs
│   ├── design_0.cif
│   └── design_0.npz
├── intermediate_designs_inverse_folded/
│   ├── refold_cif/                 # Refolded complexes
│   └── aggregate_metrics_analyze.csv
└── final_ranked_designs/
    ├── final_10_designs/           # Top designs
    └── results_overview.pdf        # Summary plots
```

**What good output looks like:**
- Refolding RMSD < 2.0A (design folds as predicted)
- ipTM > 0.5 (confident interface)
- All designs complete pipeline without errors

## Decision tree

```
Should I use BoltzGen?
│
└─ What type of design?
   ├─ All-atom precision needed → boltzgen ✓
   ├─ Ligand binding pocket → boltzgen ✓
   └─ Antibody or nanobody design  → iggm 
```

## Typical performance

| Campaign Size | Time (L40S) | Cost (Modal) | Notes |
|---------------|-------------|--------------|-------|
| 50 designs | 30-45 min | ~$8 | Quick exploration |
| 100 designs | 1-1.5h | ~$15 | Standard campaign |
| 500 designs | 5-8h | ~$70 | Large campaign |

**Per-design**: ~30-60s for typical binder.

---

## Verify

```bash
find output -name "*.cif" | wc -l  # Should match num_samples
```

---

## Troubleshooting

**Verify config first**: Always run `boltzgen check config.yaml` before running the full pipeline
**Slow generation**: Use fewer designs for initial testing, then scale up
**OOM errors**: Use A100-80GB or reduce `--num-designs`
**Wrong binding site**: Residue indices use `label_seq_id` (1-indexed), check in Molstar viewer

### Error interpretation

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: CUDA out of memory` | Large design or long protein | Use A100-80GB or reduce designs |
| `FileNotFoundError: *.cif` | Target file not found | File paths are relative to YAML location |
| `ValueError: invalid chain` | Chain not in target | Verify chain IDs with Molstar or PyMOL |
| `modal: command not found` | Modal CLI not installed | Run `pip install modal && modal setup` |

---

**Next**: Validate with `boltz-2`.