---
name: biomed-skill-creator
description: >
  Create new biomedical skills or improve existing ones for the OpenBioMed toolkit.
  Use this skill when:
  (1) Creating a new skill from scratch,
  (2) Capturing a workflow as a reusable skill,
  (3) Automating a biomedical task,
  (4) Improving an existing skill.

  This skill guides through an interactive process: define intent → design workflow →
  validate with real data → iterate → evaluate.
license: MIT
category: utilities
tags: [skill-creation, workflow-automation, meta-skill]
---

# BioMed Skill Creator

A meta-skill for creating and improving skills in the OpenBioMed biomedical toolkit.

## Overview

This skill guides you through creating biomedical skills with **interactive validation**. Biomedical workflows require domain-specific validation with real data before finalization.

**Workflow:**
```
1. Capture Intent → 2. Design Workflow → 3. Interactive Validation → 4. Finalize → 5. Evaluate
        ↑                                                              ↓
        └──────────────────── Iterate if needed ←─────────────────────┘
```

## Step 1: Capture Intent

Ask clarifying questions:
1. **What biomedical task should this skill perform?**
2. **What inputs will users provide?** (molecule name/SMILES, protein ID, text)
3. **What outputs should the skill produce?** (reports, files, predictions, visualizations)
4. **Are there edge cases or constraints?**

### Input Types

| Input Type | Factory Method | Example |
|------------|----------------|---------|
| Molecule | `Molecule.from_smiles()` | `"CC(=O)OC1=CC=CC=C1C(=O)O"` |
| Protein | `Protein.from_fasta()` | `"MKFLILLFNILCLFPVLAADNH..."` |
| Pocket | `Pocket.from_protein_ref_ligand()` | Protein + reference ligand |
| Text | `Text.from_str()` | `"What is this molecule?"` |

## Step 2: Design Workflow

Identify tools and steps. See `references/tools_reference.md` for available tools.

### Common Workflow Patterns

| Pattern | Tools Flow |
|---------|-----------|
| Drug-likeness | `molecule_name_request` → `molecule_qed/sa/logp/lipinski` → summarize |
| Protein Mutation | `protein_uniprot_request` → `mutation_explanation` → `protein_folding` → visualize |
| Structure-Based Design | `protein_pdb_request` → `extract_molecules` → `structure_based_drug_design` → docking |
| Molecule Q&A | `molecule_name_request` → `molecule_question_answering` → format |

### Basic Workflow Code Pattern

```python
from open_biomed.tools.tool_registry import TOOLS

# Get entity
tool = TOOLS["tool_name"]
result, message = tool.run(parameter=value)
entity = result.get("protein") or result.get("molecule")

# Process with other tools
another_tool = TOOLS["another_tool"]
output, msg = another_tool.run(entity=entity)
```

## Step 3: Interactive Validation (CRITICAL)

**Execute ONE step at a time and check with user before proceeding.**

After designing the workflow, ask:
> "Please provide an example input and I'll run through each step showing results."

### For Each Step

1. **Execute** the step using OpenBioMed tools
2. **Display results** with standardized format (see `references/validation_template.md`)
3. **Ask for feedback**: "Is this result satisfactory? (yes/proceed/modify/skip)"

### Handling Errors

When a step fails:
1. Explain the error clearly
2. Propose alternatives (fallback tools, web search, skip)
3. Ask user to decide

### After All Steps

Present summary and ask:
> "Do you want to:
> 1. **Proceed** with this workflow?
> 2. **Modify** and re-validate?
> 3. **Try different input**?"

## Step 4: Finalize the Skill

Once approved, create the skill files:

### Directory Structure

```
skill-name/
├── SKILL.md              # Main skill definition (< 200 lines)
├── examples/             # Runnable example scripts
│   └── basic_example.py
└── references/           # Detailed documentation
    ├── advanced.md
    └── troubleshooting.md
```

### SKILL.md Template

See `references/skill_template.md` for the full structure. Key sections:

```markdown
---
name: skill-name
description: >
  [One-line summary of what the skill does].
  Use this skill when:
  (1) [Use case 1],
  (2) [Use case 2],
  (3) [Use case 3].
license: [MIT|Apache-2.0|BSD-3-Clause|GPL-3.0|Proprietary]
category: [category from list below]
tags: [tag1, tag2, tag3]
---

# Skill Title

## When to Use
## Workflow (keep code snippets < 20 lines)
## Expected Outputs
## Error Handling
```

### License Selection

Before finalizing SKILL.md, ask the user to choose a license:

> "What license should this skill use?
> 1. **MIT** (Recommended) - Permissive, allows commercial use
> 2. **Apache-2.0** - Permissive with patent grant
> 3. **BSD-3-Clause** - Permissive, no endorsement clause
> 4. **GPL-3.0** - Copyleft, derivatives must be open source
> 5. **Proprietary** - Restricted use, not open source"

Default to **MIT** if user doesn't specify.

### Category Options

| Category | Description |
|----------|-------------|
| `drug-discovery` | Drug design, molecule generation, lead optimization, virtual screening |
| `admet-prediction` | Absorption, distribution, metabolism, excretion, toxicity prediction |
| `protein-engineering` | Protein design, stability optimization, function prediction |
| `protein-structure` | Structure prediction, folding, conformational analysis |
| `mutation-analysis` | Mutation effect prediction, variant annotation, engineering |
| `antibody-design` | Antibody/nanobody design, affinity maturation, epitope prediction |
| `immunology` | Immunogenicity prediction, vaccine design, immune profiling |
| `single-cell` | Single-cell analysis, cell annotation, spatial transcriptomics |
| `genomics` | Gene analysis, variant calling, regulatory element prediction |
| `transcriptomics` | RNA-seq analysis, expression profiling, differential expression |
| `metabolomics` | Metabolite identification, pathway analysis, metabolic modeling |
| `proteomics` | Protein identification, PTM analysis, protein-protein interactions |
| `pathway-analysis` | Pathway enrichment, network analysis, systems biology |
| `bioactivity-prediction` | Activity prediction, target identification, bioassay analysis |
| `binding-affinity` | Docking, binding prediction, protein-ligand interactions |
| `molecular-dynamics` | MD simulation, conformational sampling, free energy calculation |
| `chemical-synthesis` | Retrosynthesis, reaction prediction, synthesis planning |
| `safety-toxicology` | Toxicity prediction, safety assessment, off-target effects |
| `clinical-translational` | Biomarker discovery, patient stratification, drug repurposing |
| `bioimaging` | Medical imaging analysis, cell segmentation, image-based profiling |
| `knowledge-retrieval` | Literature mining, database queries, knowledge graphs |
| `multi-modal-reasoning` | Cross-modal tasks, text-based molecule/protein tasks, QA |
| `visualization` | Molecular visualization, structure rendering, report generation |
| `utilities` | Meta-skills, workflow automation, helper tools, evaluation |

### Writing Guidelines

1. **Keep SKILL.md under 200 lines** - Move long code to `examples/` or `references/`
2. **Code snippets < 20 lines** - Link to full examples
3. **Include interpretation** - What do scores/outputs mean?
4. **Handle errors** - What if tools/APIs fail?

## Step 5: Evaluate the Skill

Run evaluation to ensure quality. See `references/evaluation_reference.md` for details.

1. **Create 2-3 test cases** with realistic prompts
2. **Run grader** - Compare with-skill vs baseline agents
3. **Analyze results** - Identify patterns and issues
4. **Iterate** if needed

## Quick Reference

See `references/quick_reference.md` for:
- Workflow patterns summary
- Input type reference
- Score interpretation tables
- Evaluation checklist

## Communication Style

Adapt to user's familiarity:
- **Expert**: Use technical terms (ADMET, TPSA, RMSD)
- **Intermediate**: Brief explanations
- **Beginner**: Analogies, explain why metrics matter

## Checklist

Before finalizing:
- [ ] Workflow validated with real input
- [ ] User approved the workflow
- [ ] SKILL.md under 200 lines
- [ ] Long code in examples/
- [ ] Error handling documented
- [ ] Test cases created and graded
