# BioMedSkills Overview

This document provides a comprehensive overview of all available skills in the OpenBioMed toolkit.

**Last Updated:** 2026-03-16

---

## Skills Summary

| Total Skills | Verified | To Be Verified |
|--------------|----------|----------------|
| 26 | 12 | 14 |

---

## All Skills

### Biochemistry & Drug Discovery

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| drug-candidate-discovery | Drug Discovery | Generate diverse druggable molecules for a given target or disease using AI-powered drug discovery tools including target identification, structure retrieval, and molecule generation. | To Be Verified |
| drug-lead-analysis | Drug Discovery | Analyze drug candidate molecules for drug-likeness (QED, Lipinski), ADMET properties, BBB penetration, and safety profiles. | Verified |
| pocket-based-drug-design | Drug Discovery | Structure-based drug design using protein binding pockets to generate novel drug-like molecules with MolCraft model. | To Be Verified |
| admet-prediction | ADMET Prediction | Predict comprehensive ADMET properties (BBB penetration, side effects, Caco-2 permeability, half-life, LD50 toxicity) for drug candidates using GraphMVP ensemble models. | Verified |
| retrosynthesis-planning | Synthetic Chemistry | Expert-in-the-loop retrosynthetic planning workflow for breaking down target molecules into available starting materials and designing synthetic routes with AiZynthFinder integration. | Verified |

### Protein Analysis & Engineering

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| protein-mutation-analysis | Mutation Analysis | Analyze protein mutations by retrieving protein data, explaining mutation effects with MutaPLM, predicting structure with ESMFold, and visualizing results. | Verified |
| mutation-design-aav | Mutation Analysis | Design high-fitness and high-diversity mutants of AAV VP1 capsid protein through multi-round iterative optimization. | To Be Verified |
| mutation-design-gfp | Mutation Analysis | Design high-fluorescence and high-diversity GFP mutants through multi-round iterative optimization. | To Be Verified |
| functional-protein-design | Protein Engineering | Generate functional protein sequences using CodeFP with Gene Ontology (GO) tag guidance for de novo protein design. | To Be Verified |
| similar-protein-retrieval | Data Retrieval | Retrieve proteins with similar structures (FoldSeek) or sequences (MSA) from UniProt, PDB, and AFDB databases. | Verified |
| structure-prediction-boltz-2 | Structure Prediction | Predict protein complex structures and protein-ligand complexes with binding affinity (IC50) using Boltz-2. | To Be Verified |
| protein-structure-design-boltzgen | Structure Design | All-atom protein design using BoltzGen diffusion model for binder design, peptide design, and small molecule binding design. | To Be Verified |
| antibody-structure-prediction-tfold | Structure Prediction | Predict antibody/nanobody structures and antigen-antibody complex structures using tFold model. | To Be Verified |
| antibody-design-iggm | Antibody Design | Epitope-conditioned de novo antibody design and affinity maturation using IgGM model. | To Be Verified |
| binding-affinity-prediction-prodigy | Binding Analysis | Predict binding affinity scores for protein complexes using Prodigy from structure files. | To Be Verified |

### Single-Cell & Transcriptomics

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| single-cell-foundation-model-scrna-seq-geneformer | Foundation Model | Geneformer workflows for tokenization, cell/gene classification, embedding extraction, and in silico perturbation analysis. | Verified |
| single-cell-foundation-model-scrna-seq-langcell | Foundation Model | LangCell for zero-shot and few-shot cell type annotation with multimodal cell-text matching. | Verified |
| single-cell-foundation-model-scrna-seq-scgpt | Foundation Model | scGPT for preprocessing, binning, cell embedding extraction, fine-tuning, and reference mapping workflows. | Verified |
| spatial-transcriptomics-foundation-model-stofm | Foundation Model | SToFM for spatial transcriptomics preprocessing, cell embedding generation with SE(2) Transformer, and downstream analysis. | Verified |
| single-cell-scrna-seq-analysis-scanpy | Bioinformatics | Complete scRNA-seq analysis workflow with Scanpy including QC, normalization, dimensionality reduction, clustering, and marker gene identification. | To Be Verified |
| single-cell-multi-omics-analysis-scvi | Bioinformatics | Probabilistic deep learning for single-cell multi-omics analysis including scVI, scANVI, totalVI, and spatial deconvolution. | To Be Verified |
| cellxgene-census-query | Data Query | Query CZ CELLxGENE Census (61M+ cells) for single-cell expression data by cell type, tissue, or disease. | To Be Verified |
| spatial-transcriptomics-spatial-data-io | Data I/O | Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, and other platforms using Squidpy and SpatialData. | To Be Verified |

### Data Retrieval & Knowledge

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| pubchem-query | Chemical Database | Query PubChem database for chemical structures, similar compounds (similarity search), and bioactivity data against protein targets. | Verified |
| biomedical-literature-search | Literature Search | Search PubMed and bioRxiv for biomedical research papers with titles, abstracts, and metadata. | Verified |

### Utilities

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| biomed-skill-creator | Skill Development | Create new biomedical skills or improve existing ones through an interactive validation process with intent capture, workflow design, and evaluation. | Verified |

---

## Notes

- **Verified** skills contain `examples/`, `references/`, or `eval/` directories with supporting documentation and test cases.
- **To Be Verified** skills lack these directories and require additional documentation and testing.
- Skills are organized by primary category for easy navigation.
- Each skill includes a SKILL.md file with detailed workflow documentation, usage examples, and error handling guidelines.

---

## Categories Summary

| Category | Count | Skills |
|----------|-------|--------|
| Biochemistry & Drug Discovery | 5 | drug-candidate-discovery, drug-lead-analysis, pocket-based-drug-design, admet-prediction, retrosynthesis-planning |
| Protein Analysis & Engineering | 10 | protein-mutation-analysis, mutation-design-aav, mutation-design-gfp, functional-protein-design, similar-protein-retrieval, structure-prediction-boltz-2, protein-structure-design-boltzgen, antibody-structure-prediction-tfold, antibody-design-iggm, binding-affinity-prediction-prodigy |
| Single-Cell & Transcriptomics | 8 | single-cell-foundation-model-scrna-seq-geneformer, single-cell-foundation-model-scrna-seq-langcell, single-cell-foundation-model-scrna-seq-scgpt, spatial-transcriptomics-foundation-model-stofm, single-cell-scrna-seq-analysis-scanpy, single-cell-multi-omics-analysis-scvi, cellxgene-census-query, spatial-transcriptomics-spatial-data-io |
| Data Retrieval & Knowledge | 2 | pubchem-query, biomedical-literature-search |
| Utilities | 1 | biomed-skill-creator |

---

## Quick Reference by Use Case

| I want to... | Use this skill |
|--------------|----------------|
| Design new drug candidates | `drug-candidate-discovery`, `pocket-based-drug-design` |
| Analyze molecule properties | `drug-lead-analysis`, `admet-prediction` |
| Find similar compounds | `pubchem-query` |
| Analyze protein mutations | `protein-mutation-analysis`, `mutation-design-aav`, `mutation-design-gfp` |
| Predict protein structures | `structure-prediction-boltz-2`, `antibody-structure-prediction-tfold` |
| Design antibodies | `antibody-design-iggm` |
| Design proteins | `protein-structure-design-boltzgen`, `functional-protein-design` |
| Find similar proteins | `similar-protein-retrieval` |
| Predict binding affinity | `binding-affinity-prediction-prodigy` |
| Analyze scRNA-seq data | `single-cell-scrna-seq-analysis-scanpy`, `single-cell-multi-omics-analysis-scvi` |
| Use single-cell foundation models | `single-cell-foundation-model-scrna-seq-geneformer`, `single-cell-foundation-model-scrna-seq-scgpt`, `single-cell-foundation-model-scrna-seq-langcell` |
| Analyze spatial transcriptomics | `spatial-transcriptomics-foundation-model-stofm`, `spatial-transcriptomics-spatial-data-io` |
| Query single-cell databases | `cellxgene-census-query` |
| Search biomedical literature | `biomedical-literature-search` |
| Plan retrosynthesis routes | `retrosynthesis-planning` |
| Create a new skill | `biomed-skill-creator` |
