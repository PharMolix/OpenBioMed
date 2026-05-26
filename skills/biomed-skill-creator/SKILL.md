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

## Endpoint Configuration

When designing workflows that use the run_pipeline API, the base URL follows this resolution order:

1. User-provided endpoint in conversation
2. Environment variable `OPENBIOMED_API_BASE_URL`
3. Default `OPENBIOMED_CLOUD_URL = http://127.0.0.1:8092`

## Workflow

```
1. Capture Intent → 2. Design Workflow → 3. Interactive Validation → 4. Finalize → 5. Evaluate
        ↑                                                              ↓
        └──────────────────── Iterate if needed ←─────────────────────┘
```

---

## Step 1: Capture Intent

Ask clarifying questions:
1. **What biomedical task should this skill perform?**
2. **What inputs will users provide?** (molecule name/SMILES, protein ID, text)
3. **What outputs should the skill produce?** (reports, files, predictions, visualizations)
4. **Are there edge cases or constraints?**

### Input Types for API

| Input Type | API Parameter | Example |
|------------|---------------|---------|
| Molecule SMILES | `molecule` | `"CC(=O)OC1=CC=CC=C1C(=O)O"` |
| Molecule file | `molecule` | `"./tmp/molecule.pkl"` |
| Protein PDB ID | `query` | `"4AQ3"` |
| Protein file | `protein` | `"./tmp/protein.pkl"` |
| Text query | `query` | `"aspirin mechanism"` |

---

## Step 2: Design Workflow

Identify API tasks and steps.

### Common Workflow Patterns (API Style)

| Pattern | API Tasks Flow |
|---------|---------------|
| Drug-likeness | `molecule_name_request` → `molecule_property_calculation` (QED/SA/LogP/Lipinski) |
| Protein Mutation | `protein_uniprot_request` → `mutation_explanation` → `protein_folding` |
| Structure-Based Design | `protein_pdb_request` → `extract_molecules_from_pdb_file` → `create_pocket_from_ligand` → `structure_based_drug_design` → `protein_molecule_docking_score` |
| Molecule Q&A | `molecule_name_request` → `molecule_question_answering` |
| Literature Search | `literature_search` (pubmed_search) → `summarize_content` |
| Bioactivity Query | `pubchem_bioactivity` or `chembl_query` |

### API Workflow Code Pattern

```bash
# Step 1: Get molecule structure
curl -X POST "${BASE_URL}/run_pipeline/" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}'
# Response: {"molecule": "./tmp/pubchem_aspirin.pkl", "molecule_preview": "..."}

# Step 2: Calculate properties
curl -X POST "${BASE_URL}/run_pipeline/" \
  -d '{"task": "molecule_property_calculation", "molecule": "./tmp/pubchem_aspirin.pkl", "property": "QED"}'
# Response: {"score": 0.55}
```

### Available API Tasks

| Category | Tasks |
|----------|-------|
| **Molecule Operations** | `molecule_name_request`, `molecule_structure_request`, `molecule_property_calculation`, `molecule_question_answering`, `text_based_molecule_editing`, `export_molecule` |
| **Protein Operations** | `protein_uniprot_request`, `protein_pdb_request`, `protein_folding`, `protein_binding_site_prediction`, `protein_question_answering`, `protein_molecule_docking_score`, `export_protein` |
| **Drug Design** | `structure_based_drug_design`, `create_pocket_from_ligand`, `extract_molecules_from_pdb_file`, `drug_lead_analysis`, `analyze_complex_interaction` |
| **Mutation** | `mutation_explanation`, `mutation_engineering`, `apply_mutation_to_sequence` |
| **Database Query** | `pubchem_bioactivity`, `chembl_query`, `kegg_query`, `ppi_string_request`, `literature_search`, `disease_drug_intel`, `ddi_analysis`, `web_search` |
| **Visualization** | `visualize_molecule`, `visualize_protein`, `visualize_complex`, `visualize_protein_pocket` |

---

## Step 3: Interactive Validation

**Execute ONE step at a time and check with user before proceeding.**

After designing the workflow, ask:
> "Please provide an example input and I'll run through each step showing results."

### For Each Step

1. **Execute** the step via API call
2. **Display results** in standardized format
3. **Ask for feedback**: "Is this result satisfactory? (yes/proceed/modify/skip)"

### Example Validation

```
Step 1: molecule_name_request(query="aspirin")
Result: {"molecule": "./tmp/pubchem_aspirin.pkl", "molecule_preview": "CC(=O)OC1=CC=CC=C1C(=O)O"}
Status: ✅ Success

Is this result satisfactory? (yes/proceed/modify/skip)
```

### Handling Errors

When a step fails:
1. Explain the error clearly
2. Propose alternatives (fallback tasks, web search, skip)
3. Ask user to decide

### After All Steps

Present summary and ask:
> "Do you want to:
> 1. **Proceed** with this workflow?
> 2. **Modify** and re-validate?
> 3. **Try different input**?"

---

## Step 4: Finalize the Skill

Once approved, create the skill files.

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

```markdown
---
name: skill-name
description: >
  [One-line summary of what the skill does].
  Use this skill when:
  (1) [Use case 1],
  (2) [Use case 2],
  (3) [Use case 3].
license: MIT
category: [category]
tags: [tag1, tag2, tag3]
---

# Skill Title

## Endpoint Configuration

Defaults declared in this skill:
- `OPENBIOMED_CLOUD_URL = http://127.0.0.1:8092`

Resolve base URL in order: user-provided → OPENBIOMED_API_BASE_URL → default.

## Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input1` | str | Yes | ... |

## Workflow Overview

| Step | API Call | Purpose |
|------|----------|---------|
| 1 | `task_name` | ... |

## API Examples

```bash
curl -X POST "${BASE_URL}/run_pipeline/" \
  -d '{"task": "...", "query": "..."}'
```

## Expected Outputs

| Output | Description |
|--------|-------------|
| ... | ... |

## Error Handling

| Error | Solution |
|-------|----------|
| ... | ... |
```

### License Options

Default to **MIT** if user doesn't specify:
- **MIT** - Permissive, allows commercial use
- **Apache-2.0** - Permissive with patent grant
- **BSD-3-Clause** - Permissive, no endorsement clause
- **GPL-3.0** - Copyleft, derivatives must be open source

### Category Options

| Category | Description |
|----------|-------------|
| `drug-discovery` | Drug design, molecule generation, lead optimization |
| `admet-prediction` | ADMET prediction, toxicity, pharmacokinetics |
| `protein-engineering` | Protein design, stability, function prediction |
| `protein-structure` | Structure prediction, folding, conformational analysis |
| `mutation-analysis` | Mutation effects, variant annotation |
| `antibody-design` | Antibody design, affinity maturation |
| `single-cell` | Single-cell analysis, cell annotation |
| `knowledge-retrieval` | Literature mining, database queries |
| `multi-modal-reasoning` | Cross-modal tasks, QA |
| `visualization` | Molecular visualization, report generation |
| `utilities` | Meta-skills, workflow automation |

### Writing Guidelines

1. **Keep SKILL.md under 200 lines** - Move long code to `examples/`
2. **Include Endpoint Configuration** - Standard URL resolution pattern
3. **API examples < 10 lines** - Link to full workflow in examples/
4. **Include error handling** - What if API fails?

---

## Step 5: Evaluate the Skill

Run evaluation to ensure quality:

1. **Create 2-3 test cases** with realistic prompts
2. **Run validation** - Test API calls with real inputs
3. **Analyze results** - Identify patterns and issues
4. **Iterate** if needed

---

## Checklist

Before finalizing:
- [ ] Workflow validated with real API calls
- [ ] User approved the workflow
- [ ] SKILL.md under 200 lines
- [ ] Endpoint Configuration section included
- [ ] API examples in curl format
- [ ] Error handling documented
- [ ] Test cases created and validated