---
name: biomed-skill-router
description: >
  Find the most suitable skill for a given biomedical task. Use this skill when:
  (1) You are unsure which skill to use for a specific biomedical task,
  (2) You want to discover available skills for a particular domain,
  (3) You need to compare multiple skills for a given use case.
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

## How It Works

The agent analyzes the user's natural language request and determines the most suitable skill based on:
1. Understanding the user's intent and goal
2. Identifying what data/inputs the user has
3. Matching against available skill capabilities

## Available Skills

### Biochemistry & Drug Discovery

| Skill | Description |
|-------|-------------|
| drug-candidate-discovery | Generate diverse druggable molecules for a given target or disease using AI-powered drug discovery tools. |
| drug-lead-analysis | Analyze drug candidate molecules for drug-likeness (QED, Lipinski), ADMET properties, BBB penetration, and safety profiles. |
| target-based-lead-design | Generate diverse lead compounds for a specific protein target using structure-based drug design with MolCraft. |
| admet-prediction | Predict comprehensive ADMET properties (BBB penetration, side effects, Caco-2 permeability, half-life, LD50 toxicity). |
| retrosynthesis-planning | Expert-in-the-loop retrosynthetic planning workflow for breaking down target molecules into available starting materials. |
| drug-drug-interaction-analysis | Analyze potential drug-drug interactions (DDI) for up to 5 drugs using KEGG DDI database. |
| text-based-molecule-editing | Modify molecules based on natural language descriptions using MolT5/BioT5 models. |

### Protein Analysis & Engineering

| Skill | Description |
|-------|-------------|
| protein-mutation-analysis | Analyze protein mutations by retrieving protein data, explaining mutation effects with MutaPLM, and predicting structure. |
| mutation-design-aav | Design high-fitness and high-diversity mutants of AAV VP1 capsid protein. |
| mutation-design-gfp | Design high-fluorescence and high-diversity GFP mutants. |
| functional-protein-design | Generate functional protein sequences using CodeFP with Gene Ontology (GO) tag guidance. |
| similar-protein-retrieval | Retrieve proteins with similar structures or sequences from UniProt, PDB, and AFDB databases. |
| structure-prediction-boltz-2 | Predict protein complex structures and protein-ligand complexes with binding affinity using Boltz-2. |
| protein-structure-design-boltzgen | All-atom protein design using BoltzGen diffusion model for binder design and peptide design. |
| antibody-structure-prediction-tfold | Predict antibody/nanobody structures and antigen-antibody complex structures. |
| antibody-design-iggm | Epitope-conditioned de novo antibody design and affinity maturation. |
| binding-affinity-prediction-prodigy | Predict binding affinity scores for protein complexes using Prodigy. |
| protein-ligand-binding-analysis-plip | Analyze protein-ligand interactions in PDB structures using PLIP (Protein-Ligand Interaction Profiler). |
| protein-function-prediction | Predict protein function and properties from amino acid sequence using BioT5. |
| protein-subcellular-localization-prediction-biot5 | Predict protein subcellular localization from amino acid sequence using BioT5. |

### Single-Cell Omics Data Analysis

| Skill | Description |
|-------|-------------|
| single-cell-foundation-model-scrna-seq-geneformer | Geneformer workflows for tokenization, cell/gene classification, embedding extraction, and perturbation analysis. |
| single-cell-foundation-model-scrna-seq-langcell | LangCell for zero-shot and few-shot cell type annotation with multimodal cell-text matching. |
| single-cell-foundation-model-scrna-seq-scgpt | scGPT for preprocessing, binning, cell embedding extraction, fine-tuning, and reference mapping. |
| spatial-transcriptomics-foundation-model-stofm | SToFM for spatial transcriptomics preprocessing and cell embedding generation. |
| single-cell-scrna-seq-analysis-scanpy | Complete scRNA-seq analysis workflow with Scanpy including QC, normalization, clustering, and marker gene identification. |
| single-cell-multi-omics-analysis-scvi | Probabilistic deep learning for single-cell multi-omics analysis including scVI, scANVI, totalVI. |
| cellxgene-census-query | Query CZ CELLxGENE Census (61M+ cells) for single-cell expression data. |
| spatial-transcriptomics-spatial-data-io | Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, and other platforms. |
| single-cell-atac-seq-qc-processing | Trim adapters, align reads, remove duplicates, and evaluate chromatin accessibility data quality. |
| single-cell-atac-seq-peak-calling-annotaion | Call accessible chromatin peaks with MACS2 and identify differentially accessible regions. |
| single-cell-proteomics-data-processing | Load, inspect, centroid, and extract features from raw LC-MS/MS data files using pyOpenMS. |
| single-cell-proteomics-peptide-identification | Search MS2 spectra against protein databases with MSFragger/Comet. |
| single-cell-multi-omics-data-harmonization | Prepare multi-omics datasets for joint integration with normalization and batch correction. |

### Data Retrieval & Knowledge

| Skill | Description |
|-------|-------------|
| pubchem-query | Query PubChem database for chemical structures, similar compounds, and bioactivity data. |
| biomedical-literature-search | Search PubMed and bioRxiv for biomedical research papers. |
| uniprot-query | Query UniProt database for protein sequences, metadata, and search by criteria. |
| kegg-query | Query KEGG database for drug information, pathway analysis, and disease-drug-target discovery. |
| ppi-string-query | Query STRING database for protein-protein interactions with confidence scores. |
| chembl-query | Query ChEMBL database for bioactivity data on drug-like compounds. |

### Molecule Analysis

| Skill | Description |
|-------|-------------|
| iupac-name-identification-biot5 | Identify the IUPAC name of a molecule using BioT5 question answering model. |
| molecule-biochemical-significance-query-biot5 | Query a molecule's biochemical significance and roles in biology and chemistry. |

### Disease & Target Analysis

| Skill | Description |
|-------|-------------|
| disease-drug-intelligence | Comprehensive disease-to-innovative-drug analysis: targets, drugs, clinical progress, and mechanism trends. |
| target-drug-report | Generate comprehensive drug development progress reports for disease therapeutic targets. |

### Utilities

| Skill | Description |
|-------|-------------|
| biomed-skill-creator | Create new biomedical skills or improve existing ones. |

## Routing Process

When a user asks for help finding a skill:

1. **Understand the request** - Analyze what the user wants to accomplish
2. **Identify available inputs** - Determine what data the user has (molecule, protein, sequence, etc.)
3. **Match to skills** - Select the most appropriate skill(s) based on the task
4. **Provide recommendation** - Suggest the best skill and explain why

## Example Interactions

**User:** "I want to design a drug for BCL2 protein"

**Agent:** Based on your request to design drugs for the BCL2 protein, I recommend:

**`target-based-lead-design`** - This skill generates diverse lead compounds for a specific protein target using structure-based drug design. You'll need a PDB ID for BCL2. Would you like me to help you find one?

---

**User:** "What can I use to analyze my scRNA-seq data?"

**Agent:** For scRNA-seq analysis, several skills are available depending on your goal:

- **`single-cell-scrna-seq-analysis-scanpy`** - Complete analysis workflow (QC, clustering, marker genes)
- **`single-cell-foundation-model-scrna-seq-langcell`** - Cell type annotation
- **`single-cell-foundation-model-scrna-seq-geneformer`** - Embedding extraction and perturbation analysis
- **`single-cell-foundation-model-scrna-seq-scgpt`** - Embedding extraction and reference mapping

What specific analysis do you want to perform?

---

**User:** "I have a molecule and want to check if it's toxic"

**Agent:** For toxicity analysis, use **`admet-prediction`**. This skill predicts comprehensive ADMET properties including toxicity (LD50), side effects, and BBB penetration. Please provide your molecule as a SMILES string.

## See Also

- `skills_overview.md` - Full skill catalog with detailed descriptions
- `biomed-skill-creator` - Create new skills when none match your needs
