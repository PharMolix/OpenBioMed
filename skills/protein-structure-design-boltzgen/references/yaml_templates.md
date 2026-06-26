# BoltzGen YAML Configuration Templates

This document provides ready-to-use YAML templates for various BoltzGen design scenarios.

## Template 1: Protein Binder Design

**Protocol**: `protein-anything`

**Use Case**: Design a protein binder (80-140 residues) to bind a target protein at specific residues.

```yaml
entities:
  # Designed binder protein
  - protein:
      id: B                    # Chain ID for designed binder
      sequence: 80..140        # Variable length: 80-140 residues

  # Target protein from structure file
  - file:
      path: target.cif         # CIF/PDB file (relative to YAML location)
      include:                 # Which chains/residues to include
        - chain:
            id: A              # Include chain A
      binding_types:           # Specify binding site residues
        - chain:
            id: A
            binding: 45,67,89  # Binder should target these residues
```

**Required Files**: `target.cif`

---

## Template 2: Short Peptide Design

**Protocol**: `peptide-anything`

**Use Case**: Design a short peptide (12-20 residues) to bind a target protein.

```yaml
entities:
  # Designed peptide
  - protein:
      id: G                    # Chain ID for designed peptide
      sequence: 12..20         # Short peptide: 12-20 residues

  # Target protein
  - file:
      path: target.cif
      include:
        - chain:
            id: A
      binding_types:
        - chain:
            id: A
            binding: 343,344,251  # Binding site on target
      structure_groups: "all"     # Use all structural information
```

**Required Files**: `target.cif`

---

## Template 3: Small Molecule Binding

**Protocol**: `protein-small_molecule`

**Use Case**: Design a protein to bind a small molecule ligand (no CIF file needed).

```yaml
entities:
  # Designed protein
  - protein:
      id: A
      sequence: 100..150       # Medium-sized protein: 100-150 residues

  # Small molecule ligand (SMILES)
  - ligand:
      id: L                    # Ligand chain ID
      smiles: "CCO"            # SMILES string (e.g., ethanol)
      # Alternative: Use CCD database ligand
      # ccd: ATP               # ATP from CCD database
```

**Required Files**: None (ligand defined by SMILES or CCD code)

---

## Template 4: Cyclic Peptide with Disulfide Bond

**Protocol**: `peptide-anything`

**Use Case**: Design a cyclic peptide with constrained cysteines for disulfide bond.

```yaml
entities:
  # Designed peptide with cysteines
  - protein:
      id: S
      sequence: 10..14C6C3     # 10-14 designed residues + C + 6 designed + C + 3 designed

  # Target protein
  - file:
      path: target.cif
      include:
        - chain:
            id: A

# Disulfide bond constraint
constraints:
  - bond:
      atom1: [S, 11, SG]       # [chain_id, res_index, atom_name]
      atom2: [S, 18, SG]       # Disulfide between Cys11 and Cys18
```

**Required Files**: `target.cif`

---

## Template 5: WHL Stapled Peptide

**Protocol**: `peptide-anything`

**Use Case**: Design a stapled peptide using WHL (tryptophan histidine leucine) linker.

```yaml
entities:
  # Designed peptide with cysteines for stapling
  - protein:
      id: R
      sequence: 3..5C6C3       # Partial fixed with cysteines

  # WHL staple ligand
  - ligand:
      id: Q
      ccd: WHL                 # Chemical component from CCD database

  # Target protein
  - file:
      path: target.cif
      include:
        - chain:
            id: A

# Staple constraints - bond cysteines to WHL
constraints:
  - bond:
      atom1: [R, 4, SG]        # First cysteine
      atom2: [Q, 1, CK]        # WHL attachment point
  - bond:
      atom1: [R, 11, SG]       # Second cysteine
      atom2: [Q, 1, CH]        # WHL attachment point
```

**Required Files**: `target.cif`

---

## Template 6: Nanobody CDR Design

**Protocol**: `nanobody-anything`

**Use Case**: Design nanobody CDR regions while keeping framework fixed.

```yaml
entities:
  # Nanobody with framework and designed CDRs
  - protein:
      id: H
      sequence: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS
      # Framework is fixed, CDR regions can be designed using specific notation

  # Target antigen
  - file:
      path: antigen.cif
      include:
        - chain:
            id: A
```

**Required Files**: `antigen.cif`

---

## Template 7: Antibody Design

**Protocol**: `antibody-anything`

**Use Case**: Design antibody CDRs for antigen binding.

```yaml
entities:
  # Heavy chain with CDR design
  - protein:
      id: H
      sequence: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS

  # Light chain (can be fixed or designed)
  - protein:
      id: L
      sequence: DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK

  # Target antigen
  - file:
      path: antigen.cif
      include:
        - chain:
            id: A
      binding_types:
        - chain:
            id: A
            binding: 45,67,89,120,145  # Epitope residues
```

**Required Files**: `antigen.cif`

---

## Template 8: Redesign Existing Residues

**Protocol**: `protein-anything`

**Use Case**: Redesign specific residues in an existing complex.

```yaml
entities:
  - file:
      path: complex.cif         # Existing complex structure
      include:
        - chain:
            id: A
        - chain:
            id: B
      design:                   # Residues to redesign
        - chain:
            id: A
            res_index: 14..19   # Redesign residues 14-19
```

**Required Files**: `complex.cif`

---

## Template 9: Partial Target Flexibility

**Protocol**: `protein-anything`

**Use Case**: Design binder with partially flexible target regions.

```yaml
entities:
  # Designed binder
  - protein:
      id: B
      sequence: 80..120

  # Target with flexible regions
  - file:
      path: target.cif
      include:
        - chain:
            id: A
      binding_types:
        - chain:
            id: A
            binding: 45,67,89
      structure_groups:         # Define flexibility
        - group:
            visibility: 1       # 1 = fixed structure
            id: A
            res_index: 10..50
        - group:
            visibility: 0       # 0 = flexible (no structural constraint)
            id: A
            res_index: 51..60
```

**Required Files**: `target.cif`

---

## Template 10: Secondary Structure Constraints

**Protocol**: `protein-anything`

**Use Case**: Design with secondary structure constraints.

```yaml
entities:
  - file:
      path: target.cif
      include:
        - chain:
            id: A
      design:                   # Residues to redesign
        - chain:
            id: A
            res_index: 14..30
      secondary_structure:      # Constrain secondary structure
        - chain:
            id: A
            helix: 15..22       # Residues 15-22 should be helix
            sheet: 28,29        # Residues 28-29 should be sheet
            loop: 14,23..27,30  # Loop regions
```

**Required Files**: `target.cif`

---

## Template 11: Not-Binding Regions

**Protocol**: `protein-anything`

**Use Case**: Design binder that avoids binding to specific regions.

```yaml
entities:
  # Designed binder
  - protein:
      id: B
      sequence: 80..140

  # Target with specified binding and non-binding regions
  - file:
      path: target.cif
      include:
        - chain:
            id: A
        - chain:
            id: B               # Include second chain
      binding_types:
        - chain:
            id: A
            binding: 45,67,89   # Should bind here
        - chain:
            id: B
            not_binding: "all"  # Should NOT bind to chain B
```

**Required Files**: `target.cif`

---

## Template 12: Fixed Protein in Complex

**Protocol**: `protein-anything`

**Use Case**: Include a fixed (non-designed) protein in the complex.

```yaml
entities:
  # Designed binder
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

  # Non-designed protein (fixed sequence)
  - protein:
      id: X
      sequence: AAVTTTTPPP      # Fixed sequence, will not be redesigned
```

**Required Files**: `target.cif`

---

## Notes

- **Residue indices**: Use `label_seq_id` (1-indexed), not `auth_seq_id`
- **File paths**: Relative to YAML file location
- **Sequence formats**: See `parameters_guide.md` for detailed format specifications
- **Validation**: Run `boltzgen check config.yaml` before full pipeline