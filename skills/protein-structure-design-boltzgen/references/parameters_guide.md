# BoltzGen YAML Configuration Parameters Guide

This document provides comprehensive parameter documentation and conversational guidance for BoltzGen YAML configuration files.

## Table of Contents

1. [Conversational Configuration Guide](#1-conversational-configuration-guide)
2. [API Layer Parameters](#2-api-layer-parameters)
3. [Design Protocols](#3-design-protocols)
4. [Entity Types](#4-entity-types)
5. [Sequence Specification Formats](#5-sequence-specification-formats)
6. [File Entity Parameters](#6-file-entity-parameters)
7. [Ligand Entity Parameters](#7-ligand-entity-parameters)
8. [Constraint Configuration](#8-constraint-configuration)
9. [Advanced Options](#9-advanced-options)
10. [Residue Index Formats](#10-residue-index-formats)
11. [Best Practices](#11-best-practices)

---

## 1. Conversational Configuration Guide

**DO NOT ask user to upload YAML file.** Instead, guide the conversation to collect design parameters systematically.

### 1.1 Protocol Selection Question

Ask user to choose based on their target:

| Protocol | Use Case | Typical Runtime |
|----------|----------|-----------------|
| `protein-anything` | Design proteins binding proteins/peptides | 15-45 min |
| `peptide-anything` | Design cyclic peptides binding proteins | 12-30 min |
| `protein-small_molecule` | Design proteins binding small molecules | 20-40 min |
| `nanobody-anything` | Design nanobody CDR loops | 15-35 min |
| `antibody-anything` | Design antibody heavy/light chain CDRs | 25-50 min |

**Question template**: "请选择设计协议：设计蛋白结合蛋白用 protein-anything，环肽用 peptide-anything，结合小分子用 protein-small_molecule，纳米抗体用 nanobody-anything，抗体用 antibody-anything"

### 1.2 Questions by Protocol Type

#### For `protein-anything` / `peptide-anything` (protein target)

| Question | Parameter | Example Response |
|----------|-----------|------------------|
| 1. 目标蛋白的CIF/PDB文件路径？ | `file.path` | `/path/to/target.cif` |
| 2. 目标蛋白的链ID？ | `chain.id` | `A` |
| 3. 希望结合的目标残基位置？ | `binding` | `45,67,89` |
| 4. 设计蛋白的链ID？ | `protein.id` | `B` |
| 5. 设计蛋白的长度范围？ | `sequence` | `80..140` 或 `100` |
| 6. 是否需要二硫键约束？(可选) | `constraints` | `11和18位的Cys` |

**Question sequence**:
```
1. "目标蛋白的CIF/PDB文件路径是什么？（服务器上的路径）"
2. "目标蛋白的链ID是什么？（如 A, B）"
3. "希望设计的蛋白结合在目标蛋白的哪些残基位置？（如 45,67,89，使用label_seq_id编号）"
4. "设计蛋白的链ID用什么？（如 B）"
5. "设计蛋白的长度范围是多少？（如 80..140 表示80-140残基，或 100 表示固定100残基）"
6. "是否需要二硫键约束？如有，请提供两个Cys的残基位置（如 11和18）"
```

#### For `protein-small_molecule` (small molecule target)

| Question | Parameter | Example Response |
|----------|-----------|------------------|
| 1. 小分子的SMILES或CCD代码？ | `smiles` / `ccd` | `c1ccccc1` 或 `ATP` |
| 2. 配体的链ID？ | `ligand.id` | `L` |
| 3. 设计蛋白的链ID？ | `protein.id` | `A` |
| 4. 设计蛋白的长度范围？ | `sequence` | `100..150` |

**Question sequence**:
```
1. "小分子的SMILES是什么？（如 c1ccccc1 代表苯）或者使用CCD数据库代码？（如 ATP, NAG）"
2. "配体的链ID用什么？（如 L）"
3. "设计蛋白的链ID用什么？（如 A）"
4. "设计蛋白的长度范围？（如 100..150）"
```

#### For `nanobody-anything` / `antibody-anything`

| Question | Parameter | Example Response |
|----------|-----------|------------------|
| 1. 抗体/纳米抗体的框架序列？ | `sequence` | `EVQLVES...` |
| 2. 重链链ID？ | `protein.id` | `H` |
| 3. 轻链序列和ID？（仅抗体） | `sequence`, `id` | `L: DIQMTQ...` |
| 4. 抗原的CIF/PDB文件路径？ | `file.path` | `/path/to/antigen.cif` |
| 5. 抗原链ID？ | `chain.id` | `A` |
| 6. 表位残基位置？（可选） | `binding` | `45,67,89` |

**Question sequence**:
```
1. "请提供抗体/纳米抗体的框架序列（完整的氨基酸序列）"
2. "重链ID用什么？（如 H）"
   - (仅抗体) "轻链ID和序列？（如 L: DIQMTQSPSS...）"
3. "抗原的CIF/PDB文件路径？"
4. "抗原的链ID？（如 A）"
5. "抗原上的表位残基位置？（可选，如 45,67,89）"
```

### 1.3 Design Parameter Questions (for all protocols)

| Question | Parameter | Default | Example |
|----------|-----------|---------|---------|
| 设计数量？ | `num_designs` | `10` | 测试10，生产10000-60000 |
| 最终输出数量？ | `budget` | `2` | 2-10 |

**Question template**: "设计数量？测试用10，生产环境推荐10000-60000。最终输出数量？默认2"

### 1.4 Example Conversation Flows

#### Example 1: Protein Binder Design

**User**: "我想设计一个蛋白结合到某个目标蛋白"

**Agent**:
```
1. "请选择设计协议：protein-anything（蛋白结合蛋白）"
2. "目标蛋白的CIF/PDB文件路径是什么？"
   → User: "/data/target.cif"
3. "目标蛋白的链ID？（如 A）"
   → User: "A"
4. "希望结合的目标残基位置？（使用label_seq_id编号）"
   → User: "45,67,89"
5. "设计蛋白的链ID？（如 B）"
   → User: "B"
6. "设计蛋白的长度范围？（如 80..140）"
   → User: "80..120"
7. "设计数量？测试用10，生产环境推荐10000-60000"
   → User: "10"
8. [生成YAML] → [上传文件] → [提交任务]
```

#### Example 2: Small Molecule Binding

**User**: "我想设计一个蛋白结合小分子苯"

**Agent**:
```
1. "协议是 protein-small_molecule"
2. "苯的SMILES是 c1ccccc1，确认使用这个吗？"
   → User: "确认"
3. "配体链ID？（如 L）"
   → User: "L"
4. "设计蛋白链ID？（如 A）"
   → User: "A"
5. "设计蛋白长度范围？（如 100..150）"
   → User: "100..150"
6. [生成YAML] → [上传YAML] → [提交任务]
```

### 1.5 YAML Generation Templates

Based on collected parameters, generate YAML using templates from `yaml_templates.md`.

Refer to: `references/yaml_templates.md` for complete template library.

---

## 1. API Layer Parameters

These parameters are passed to `/run_pipeline/` API endpoint, not in YAML file.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `boltzgen_yaml_file` | string | **Yes** | — | Design YAML file server path |
| `boltzgen_protocol` | string | No | `protein-anything` | Design protocol |
| `boltzgen_num_designs` | int | No | `10` | Intermediate designs count (production: 10000-60000) |
| `boltzgen_budget` | int | No | `2` | Final diversity-optimized set size |
| `boltzgen_cif_files` | list | No | — | CIF/PDB target file server paths |
| `boltzgen_output_name` | string | No | auto-generated | Output file name prefix |

### Parameter Guidelines

- **`num_designs`**: 
  - Test/validation: `10-100`
  - Production: `10000-60000`
  - Higher values = more diverse candidates but longer runtime

- **`budget`**: 
  - Controls final output count after diversity optimization
  - Typically `2-10` for practical use

---

## 2. Design Protocols

| Protocol | Use Case | Expected Runtime |
|----------|----------|------------------|
| `protein-anything` | Design proteins binding proteins/peptides | 15-45 min |
| `peptide-anything` | Design cyclic peptides binding proteins | 12-30 min |
| `protein-small_molecule` | Design proteins binding small molecules | 20-40 min |
| `nanobody-anything` | Design nanobody CDR loops | 15-35 min |
| `antibody-anything` | Design antibody heavy/light chain CDRs | 25-50 min |

### Protocol Selection Guide

```
Target Type?
├─ Protein/Peptide → protein-anything
├─ Small molecule → protein-small_molecule
├─ Nanobody → nanobody-anything
└─ Antibody → antibody-anything

Design Size?
├─ < 30 residues → peptide-anything (cyclic peptide)
├─ 30-200 residues → protein-anything
└─ Antibody framework → antibody-anything / nanobody-anything
```

---

## 3. Entity Types

### 3.1 `protein` Entity

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | char | **Yes** | Chain identifier (single letter) |
| `sequence` | string | **Yes** | Sequence specification (multiple formats supported) |

**Chain ID Rules**:
- Single uppercase letter (A-Z)
- Must be unique across all entities
- Common conventions: `A`, `B` for targets; `H`, `L` for antibody chains

### 3.2 `file` Entity

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | **Yes** | CIF/PDB file path (relative to YAML) |
| `include` | list | **Yes** | Chains/residues to include |
| `binding_types` | list | No | Binding site specification |
| `structure_groups` | list | No | Flexibility control |
| `design` | list | No | Residues to redesign |
| `secondary_structure` | list | No | SS constraints |

### 3.3 `ligand` Entity

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | char | **Yes** | Ligand chain identifier |
| `smiles` | string | Conditional | SMILES representation |
| `ccd` | string | Conditional | CCD database code (e.g., `ATP`) |

**Note**: Either `smiles` or `ccd` must be specified.

---

## 4. Sequence Specification Formats

### Format Types

| Format | Meaning | Example | Use Case |
|--------|---------|---------|----------|
| `N..M` | Random length range | `80..140` | Variable-length binder |
| `N` | Exact length | `80` | Fixed-length design |
| `SEQUENCE` | Fixed amino acids | `EVQLVES...` | Known framework |
| `N..MSEQUENCE` | Mixed design+fixed | `10C6C3` | Cyclic peptide with Cys |
| `FIXEDN..MFIXED` | Fixed + design + fixed | `AAAVVV20PPP` | Partially constrained |

### Detailed Format Specification

#### 4.1 Length Range (`N..M`)

```yaml
sequence: 80..140    # Design 80-140 residues (length optimized by diffusion)
sequence: 100        # Exactly 100 residues
```

#### 4.2 Fixed Sequence

```yaml
sequence: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS
```

#### 4.3 Mixed Format (Design + Fixed)

```yaml
# Design 10-14 residues, then Cys, then 6 designed, then Cys, then 3 designed
sequence: 10..14C6C3

# Fixed prefix, designed middle, fixed suffix
sequence: AAAVVV20PPP    # AAA + 20 designed + PPP
```

### Amino Acid Codes

| Code | Amino Acid | Code | Amino Acid |
|------|------------|------|------------|
| A | Alanine | R | Arginine |
| N | Asparagine | D | Aspartate |
| C | Cysteine | Q | Glutamine |
| E | Glutamate | G | Glycine |
| H | Histidine | I | Isoleucine |
| L | Leucine | K | Lysine |
| M | Methionine | F | Phenylalanine |
| P | Proline | S | Serine |
| T | Threonine | W | Tryptophan |
| Y | Tyrosine | V | Valine |

---

## 5. File Entity Parameters

### 5.1 `include` Parameter

Specifies which chains/residues from structure file to include.

```yaml
include:
  - chain:
      id: A              # Include entire chain A
  - chain:
      id: B
      res_index: 10..50  # Include only residues 10-50 from chain B
```

**Sub-parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `chain.id` | char | Chain identifier from CIF/PDB |
| `chain.res_index` | string | Residue range to include |

### 5.2 `binding_types` Parameter

Specifies where designed binder should target.

```yaml
binding_types:
  - chain:
      id: A
      binding: 45,67,89      # Positive binding (binder should contact)
      not_binding: "all"     # Negative binding (binder should avoid)
```

**Binding specification modes**:

| Mode | Syntax | Effect |
|------|--------|--------|
| Positive | `binding: 45,67,89` | Binder targets these residues |
| Negative | `not_binding: "all"` | Binder avoids this chain entirely |
| Negative specific | `not_binding: 10..20` | Binder avoids these residues |

### 5.3 `structure_groups` Parameter

Controls flexibility of target regions during diffusion.

```yaml
structure_groups:
  - group:
      visibility: 1           # 1 = fixed (structural constraint applied)
      id: A
      res_index: 10..50
  - group:
      visibility: 0           # 0 = flexible (no structural constraint)
      id: A
      res_index: 51..60
```

**Visibility values**:

| Value | Meaning |
|-------|---------|
| `1` | Fixed - target structure provides strong constraint |
| `0` | Flexible - target can move during diffusion |

### 5.4 `design` Parameter

Specifies residues on target to redesign (inverse folding mode).

```yaml
design:
  - chain:
      id: A
      res_index: 14..19       # Redesign residues 14-19 while keeping structure
```

### 5.5 `secondary_structure` Parameter

Constrains secondary structure of designed/re designed regions.

```yaml
secondary_structure:
  - chain:
      id: A
      helix: 15..22           # Residues 15-22 should be α-helix
      sheet: 28,29            # Residues 28-29 should be β-sheet
      loop: 14,23..27,30      # Loop regions
```

**SS types**:

| Type | Syntax | Description |
|------|--------|-------------|
| `helix` | `15..22` or `15,16,17` | α-helix constraint |
| `sheet` | `28,29` | β-sheet constraint |
| `loop` | `14,23..27,30` | Loop/coil constraint |

---

## 6. Ligand Entity Parameters

### 6.1 SMILES Format

```yaml
ligand:
  id: L
  smiles: "CCO"                # Ethanol
  smiles: "c1ccccc1"           # Benzene
  smiles: "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
```

### 6.2 CCD Database Code

```yaml
ligand:
  id: L
  ccd: ATP                     # Adenosine triphosphate
  ccd: WHL                     # WHL staple linker
  ccd: NAG                     # N-acetylglucosamine
```

**Common CCD codes**:

| Code | Ligand | Code | Ligand |
|------|--------|------|--------|
| ATP | Adenosine triphosphate | ADP | Adenosine diphosphate |
| NAG | N-acetylglucosamine | MAN | Mannose |
| WHL | Staple linker | ZN | Zinc ion |

---

## 7. Constraint Configuration

### 7.1 Bond Constraints

Define covalent bonds between atoms.

```yaml
constraints:
  - bond:
      atom1: [S, 11, SG]       # [chain_id, residue_index, atom_name]
      atom2: [S, 18, SG]       # Disulfide bond
```

**Atom specification format**: `[chain_id, residue_index, atom_name]`

| Field | Type | Description |
|-------|------|-------------|
| chain_id | char | Chain identifier |
| residue_index | int | Residue number (1-indexed) |
| atom_name | string | Atom name (e.g., `SG` for Cys sulfur) |

### 7.2 Common Bond Types

#### Disulfide Bond
```yaml
constraints:
  - bond:
      atom1: [S, 11, SG]       # Cys11 sulfur
      atom2: [S, 18, SG]       # Cys18 sulfur
```

#### Staple Bond (WHL)
```yaml
constraints:
  - bond:
      atom1: [R, 4, SG]        # Cysteine SG
      atom2: [Q, 1, CK]        # WHL attachment point
  - bond:
      atom1: [R, 11, SG]       # Second cysteine
      atom2: [Q, 1, CH]        # WHL second attachment
```

### 7.3 Atom Name Reference

| Residue | Atom | Name | Residue | Atom | Name |
|---------|------|------|---------|------|------|
| Cysteine | Sulfur | SG | Histidine | Nitrogen | ND1, NE2 |
| Lysine | Nitrogen | NZ | Aspartate | Oxygen | OD1, OD2 |
| Serine | Oxygen | OG | Threonine | Oxygen | OG1 |

---

## 8. Advanced Options

### 8.1 Multiple Binding Sites

```yaml
binding_types:
  - chain:
      id: A
      binding: 45,67,89        # Primary binding site
  - chain:
      id: B
      binding: 120,145         # Secondary binding site
```

### 8.2 Complex Include Patterns

```yaml
include:
  - chain:
      id: A
      res_index: 1..100        # Partial chain
  - chain:
      id: B                    # Full chain
  - chain:
      id: C
      res_index: 50..80        # Another partial
```

### 8.3 Multiple Design Regions

```yaml
design:
  - chain:
      id: A
      res_index: 14..19        # First redesign region
  - chain:
      id: A
      res_index: 45..50        # Second redesign region
```

### 8.4 Mixed Visibility Groups

```yaml
structure_groups:
  - group:
      visibility: 1            # Fixed backbone
      id: A
      res_index: 1..100
  - group:
      visibility: 0            # Flexible side chains
      id: A
      res_index: 45..55
```

---

## 9. Residue Index Formats

All residue indices use `label_seq_id` (1-indexed) from CIF/PDB files.

### Format Variants

| Format | Example | Meaning |
|--------|---------|---------|
| Single | `45` | Residue 45 only |
| Range | `14..19` | Residues 14 to 19 inclusive |
| Multiple | `45,67,89` | Residues 45, 67, and 89 |
| Mixed | `14,23..27,30` | 14, 23-27, and 30 |

### Important Notes

- **Use `label_seq_id`**, not `auth_seq_id`
- CIF files may have different `label_seq_id` vs `auth_seq_id`
- Verify residue numbers in molecular viewer (Molstar, PyMOL)
- Check CIF file header for numbering scheme

### Finding Correct Indices

```bash
# Extract label_seq_id from CIF
grep "^ATOM" target.cif | awk '{print $7}' | sort -u

# Or use Python
from Bio.PDB import MMCIFParser
parser = MMCIFParser()
structure = parser.get_structure("target", "target.cif")
for residue in structure[0]['A']:
    print(residue.id[1])  # label_seq_id
```

---

## 10. Best Practices

### 10.1 File Preparation

1. Use CIF format (preferred) or PDB
2. Verify chain IDs match CIF file
3. Check residue numbering scheme
4. Ensure structure is complete (no missing residues at binding site)

### 10.2 YAML Validation

```bash
# Syntax validation
boltzgen check config.yaml

# Full structure validation
boltzgen validate config.yaml --target target.cif
```

### 10.3 Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong residue indices | `ValueError: invalid chain` | Use `label_seq_id` |
| Missing CIF file | `FileNotFoundError` | Check file path |
| Wrong chain ID | Binding to wrong region | Verify in CIF file |
| Invalid SMILES | Ligand parse error | Validate SMILES string |

### 10.4 Performance Tips

- Start with small `num_designs` (10-100) for testing
- Use appropriate sequence length range
- Specify binding sites precisely
- Include relevant chains only

### 10.5 Output Interpretation

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| Refolding RMSD | < 2.0 Å | Design folds correctly |
| ipTM | > 0.5 | Confident interface prediction |
| pAE | < 10 | Low alignment error |

---

## Quick Reference Card

### Entity Declaration

```yaml
entities:
  - protein: {id: B, sequence: 80..140}
  - file: {path: target.cif, include: [{chain: {id: A}}]}
  - ligand: {id: L, smiles: "CCO"}
```

### Binding Specification

```yaml
binding_types:
  - chain: {id: A, binding: 45,67,89}
  - chain: {id: B, not_binding: "all"}
```

### Constraints

```yaml
constraints:
  - bond: {atom1: [S, 11, SG], atom2: [S, 18, SG]}
```

### Residue Ranges

- Single: `45`
- Range: `14..19`
- Multiple: `45,67,89`
- Mixed: `14,23..27,30`