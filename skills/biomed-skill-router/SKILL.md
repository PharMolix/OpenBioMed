---
name: biomed-skill-router
description: >
  Find the most suitable skill for a given biomedical task. Use this skill when:
  (1) You are unsure which skill to use for a specific biomedical task,
  (2) You want to discover available skills for a particular domain,
  (3) You need to compare multiple skills for a given use case.

  This skill routes user requests to the appropriate OpenBioMed skill based on
  natural language analysis of the task description.
license: MIT
category: utilities
tags: [skill-discovery, routing, recommendation, navigation]
---

# BioMed Skill Router

Find the most suitable skill for your biomedical task by analyzing your request and matching it against available skills.

## When to Use

- User describes a biomedical task but doesn't know which skill to use
- User wants to discover what skills are available for a particular domain
- User needs help choosing between multiple skills for a similar task
- User asks "What can I do with OpenBioMed?" or "Which skill should I use for X?"

## Routing Process

When a user asks for help finding a skill:

1. **Understand the request** - Analyze what the user wants to accomplish
2. **Identify available inputs** - Determine what data the user has (molecule, protein, sequence, etc.)
3. **Match to skills** - Select the most appropriate skill(s) based on the task
4. **Provide recommendation** - Suggest the best skill and explain why

---

## Available Skills by Category

### Biochemistry & Drug Discovery

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| drug-candidate-discovery | Generate diverse druggable molecules for a target/disease | `web_search`, `protein_pdb_request`, `extract_molecules_from_pdb_file`, `create_pocket_from_ligand`, `structure_based_drug_design`, `molecule_property_calculation` |
| drug-lead-analysis | Analyze drug candidate molecules for drug-likeness, ADMET, safety | `molecule_property_calculation`, `drug_lead_analysis` |
| target-based-lead-design | Generate lead compounds for a specific protein target | `protein_pdb_request`, `extract_molecules_from_pdb_file`, `create_pocket_from_ligand`, `structure_based_drug_design`, `protein_molecule_docking_score`, `analyze_complex_interaction` |
| admet-prediction | Predict ADMET properties (BBB, side effects, toxicity) | `molecule_property_prediction`, `molecule_property_calculation` |
| retrosynthesis-planning | Retrosynthetic planning for target molecules | `retrosynthesis` |
| drug-drug-interaction-analysis | Analyze drug-drug interactions for multiple drugs | `ddi_analysis` |
| text-based-molecule-editing | Modify molecules based on natural language descriptions | `text_based_molecule_editing` |

### Protein Analysis & Engineering

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| protein-mutation-analysis | Analyze protein mutations and predict effects | `protein_uniprot_request`, `mutation_explanation`, `protein_folding` |
| mutation-design-aav | Design AAV VP1 capsid mutants | `mutation_engineering` |
| mutation-design-gfp | Design GFP mutants | `mutation_engineering` |
| functional-protein-design | Generate functional protein sequences with GO guidance | `go_guided_protein_generation` |
| similar-protein-retrieval | Retrieve similar proteins from databases | `protein_uniprot_request`, `ppi_string_request` |
| structure-prediction-boltz-2 | Predict protein complex structures | `protein_folding` |
| protein-structure-design-boltzgen | All-atom protein design | `protein_structure_design` |
| antibody-structure-prediction-tfold | Predict antibody structures | `protein_folding` |
| antibody-design-iggm | Epitope-conditioned antibody design | `antibody_design` |
| binding-affinity-prediction-prodigy | Predict binding affinity for complexes | `binding_affinity_prediction` |
| protein-ligand-binding-analysis-plip | Analyze protein-ligand interactions | `analyze_complex_interaction` |
| protein-function-prediction | Predict protein function from sequence | `protein_question_answering` |
| protein-subcellular-localization-prediction-biot5 | Predict subcellular localization | `protein_question_answering` |

### Single-Cell Omics Data Analysis

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| single-cell-foundation-model-scrna-seq-geneformer | Geneformer workflows | `cell_annotation` |
| single-cell-foundation-model-scrna-seq-langcell | LangCell cell type annotation | `cell_annotation` |
| single-cell-foundation-model-scrna-seq-scgpt | scGPT embeddings and mapping | `cell_annotation` |
| spatial-transcriptomics-foundation-model-stofm | SToFM spatial embeddings | `spatial_transcriptomics` |
| single-cell-scrna-seq-analysis-scanpy | Complete scRNA-seq analysis | Local processing |
| single-cell-multi-omics-analysis-scvi | Multi-omics integration | Local processing |
| cellxgene-census-query | Query CELLxGENE Census | `cellxgene_query` |
| spatial-transcriptomics-spatial-data-io | Load spatial transcriptomics data | Local processing |
| single-cell-atac-seq-qc-processing | ATAC-seq QC and processing | Local processing |
| single-cell-atac-seq-peak-calling-annotaion | Peak calling and annotation | Local processing |
| single-cell-proteomics-data-processing | LC-MS/MS data processing | Local processing |
| single-cell-proteomics-peptide-identification | Peptide identification | Local processing |

### Data Retrieval & Knowledge

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| pubchem-query | Query PubChem for compounds and bioactivity | `molecule_name_request`, `molecule_structure_request`, `pubchem_bioactivity` |
| biomedical-literature-search | Search PubMed and bioRxiv | `literature_search` |
| uniprot-query | Query UniProt for proteins | `protein_uniprot_request` |
| kegg-query | Query KEGG for drugs, pathways | `kegg_query` |
| ppi-string-query | Query STRING for PPI | `ppi_string_request` |
| chembl-query | Query ChEMBL for bioactivity | `chembl_query`, `disease_drug_intel` |

### Molecule Analysis

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| iupac-name-identification-biot5 | Identify IUPAC name of molecule | `molecule_question_answering` |
| molecule-biochemical-significance-query-biot5 | Query molecule biochemical significance | `molecule_question_answering` |

### Disease & Target Analysis

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| disease-drug-intelligence | Disease-to-drug analysis | `disease_drug_intel` |
| target-drug-report | Drug development progress reports | `web_search`, `literature_search`, `disease_drug_intel` |

### Utilities

| Skill | Description | API Tasks Used |
|-------|-------------|----------------|
| biomed-skill-creator | Create new biomedical skills | Local processing |

---

## Example Interactions

### Drug Design Request

**User:** "I want to design a drug for BCL2 protein"

**Recommendation:**
```
Based on your request to design drugs for BCL2 protein, recommend:

**target-based-lead-design**
- Generates lead compounds for specific protein targets
- Uses structure-based drug design with MolCraft
- Requires: PDB ID for BCL2 (e.g., 4AQ3, 4LVT)

API workflow:
1. protein_pdb_request → Get BCL2 structure
2. extract_molecules_from_pdb_file → Extract protein and ligands
3. create_pocket_from_ligand → Define binding pocket
4. structure_based_drug_design → Generate molecules
```

---

### scRNA-seq Analysis Request

**User:** "What can I use to analyze my scRNA-seq data?"

**Recommendation:**
```
For scRNA-seq analysis, recommend based on your goal:

| Goal | Recommended Skill |
|------|------------------|
| Complete workflow | single-cell-scrna-seq-analysis-scanpy |
| Cell type annotation | single-cell-foundation-model-scrna-seq-langcell |
| Embedding extraction | single-cell-foundation-model-scrna-seq-geneformer |
| Reference mapping | single-cell-foundation-model-scrna-seq-scgpt |

What specific analysis do you want?
```

---

### Toxicity Analysis Request

**User:** "I have a molecule and want to check if it's toxic"

**Recommendation:**
```
For toxicity analysis, recommend:

**admet-prediction**
- Predicts comprehensive ADMET properties
- Includes: LD50 toxicity, side effects, BBB penetration
- API: molecule_property_prediction, molecule_property_calculation

**drug-lead-analysis**
- Comprehensive drug-likeness analysis
- Includes: QED, Lipinski, safety profiles
- API: drug_lead_analysis

Provide your molecule as SMILES string.
```

---

### Database Query Request

**User:** "Find information about aspirin in PubChem"

**Recommendation:**
```
For PubChem queries, use:

**pubchem-query**
- Name to structure: molecule_name_request(query="aspirin")
- Similarity search: molecule_structure_request
- Bioactivity: pubchem_bioactivity

API call:
curl -X POST "${BASE_URL}/run_pipeline/" \
  -d '{"task": "molecule_name_request", "query": "aspirin"}'
```

---

## Task to Skill Mapping

| User Task | Recommended Skill | Primary API Task |
|-----------|------------------|------------------|
| Design drugs for target | target-based-lead-design | structure_based_drug_design |
| Analyze molecule properties | drug-lead-analysis | molecule_property_calculation |
| Find similar compounds | pubchem-query | molecule_structure_request |
| Predict protein structure | structure-prediction-boltz-2 | protein_folding |
| Analyze mutations | protein-mutation-analysis | mutation_explanation |
| Search literature | biomedical-literature-search | literature_search |
| Query protein interactions | ppi-string-query | ppi_string_request |
| Drug-drug interactions | drug-drug-interaction-analysis | ddi_analysis |

---

## See Also

- All available skills listed in `.claude/skills/` directory
- `biomed-skill-creator` - Create new skills when none match your needs