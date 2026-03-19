# BioMedSkills Overview

This document provides a comprehensive overview of all available skills in the OpenBioMed toolkit.

**Last Updated:** 2026-03-19

---

## Skills Summary

| Total Skills |
|--------------|
| 41 |

---

## All Skills

### Biochemistry & Drug Discovery

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| drug-candidate-discovery | Drug Discovery | Generate diverse druggable molecules for a given target or disease using AI-powered drug discovery tools including target identification, structure retrieval, and molecule generation. | |
| drug-lead-analysis | Drug Discovery | Analyze drug candidate molecules for drug-likeness (QED, Lipinski), ADMET properties, BBB penetration, and safety profiles. | Recommended |
| target-based-lead-design | Drug Discovery | Generate diverse lead compounds for a specific protein target using structure-based drug design with MolCraft. Includes docking, ADMET filtering, diversity selection, and iterative refinement. | Recommended |
| admet-prediction | ADMET Prediction | Predict comprehensive ADMET properties (BBB penetration, side effects, Caco-2 permeability, half-life, LD50 toxicity) for drug candidates using GraphMVP ensemble models. | Recommended |
| retrosynthesis-planning | Synthetic Chemistry | Expert-in-the-loop retrosynthetic planning workflow for breaking down target molecules into available starting materials and designing synthetic routes with AiZynthFinder integration. | Recommended |
| iupac-name-identification-biot5 | Drug Discovery | Identify the IUPAC name of a molecule using BioT5 question answering model. | |
| molecule-biochemical-significance-query-biot5 | Multi-Modal Reasoning | Query a molecule's biochemical significance and roles in biology and chemistry using BioT5 multi-modal model. | |
| text-based-molecule-editing | Drug Discovery | Modify molecules based on natural language descriptions using MolT5/BioT5 models for property optimization (solubility, potency, drug-likeness). | |

### Protein Analysis & Engineering

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| protein-mutation-analysis | Mutation Analysis | Analyze protein mutations by retrieving protein data, explaining mutation effects with MutaPLM, predicting structure with ESMFold, and visualizing results. | Recommended |
| mutation-design-aav | Mutation Analysis | Design high-fitness and high-diversity mutants of AAV VP1 capsid protein through multi-round iterative optimization. | |
| mutation-design-gfp | Mutation Analysis | Design high-fluorescence and high-diversity GFP mutants through multi-round iterative optimization. | |
| functional-protein-design | Protein Engineering | Generate functional protein sequences using CodeFun with Gene Ontology (GO) tag guidance for de novo protein design. | |
| protein-function-prediction | Protein Engineering | Predict protein function and properties from amino acid sequences using BioT5 for functional annotation and pathway analysis. | |
| similar-protein-retrieval | Data Retrieval | Retrieve proteins with similar structures (FoldSeek) or sequences (MSA) from UniProt, PDB, and AFDB databases. | Recommended |
| structure-prediction-boltz-2 | Structure Prediction | Predict protein complex structures and protein-ligand complexes with binding affinity (IC50) using Boltz-2. | |
| protein-structure-design-boltzgen | Structure Design | All-atom protein design using BoltzGen diffusion model for binder design, peptide design, and small molecule binding design. | |
| antibody-structure-prediction-tfold | Structure Prediction | Predict antibody/nanobody structures and antigen-antibody complex structures using tFold model. | |
| antibody-design-iggm | Antibody Design | Epitope-conditioned de novo antibody design and affinity maturation using IgGM model. | |
| binding-affinity-prediction-prodigy | Binding Analysis | Predict binding affinity scores for protein complexes using Prodigy from structure files. | Recommended |
| protein-ligand-binding-analysis-plip | Binding Analysis | Analyze protein-ligand interactions in PDB structures using PLIP for hydrogen bonds, hydrophobic contacts, π-stacking, salt bridges, and visualization. | |

### Single-Cell Omics Data Analysis

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| single-cell-foundation-model-scrna-seq-geneformer | Foundation Model | Geneformer workflows for tokenization, cell/gene classification, embedding extraction, and in silico perturbation analysis. | Recommended |
| single-cell-foundation-model-scrna-seq-langcell | Foundation Model | LangCell for zero-shot and few-shot cell type annotation with multimodal cell-text matching. | Recommended |
| single-cell-foundation-model-scrna-seq-scgpt | Foundation Model | scGPT for preprocessing, binning, cell embedding extraction, fine-tuning, and reference mapping workflows. | Recommended |
| spatial-transcriptomics-foundation-model-stofm | Foundation Model | SToFM for spatial transcriptomics preprocessing, cell embedding generation with SE(2) Transformer, and downstream analysis. | Recommended |
| single-cell-scrna-seq-analysis-scanpy | Bioinformatics | Complete scRNA-seq analysis workflow with Scanpy including QC, normalization, dimensionality reduction, clustering, and marker gene identification. | |
| single-cell-multi-omics-analysis-scvi | Bioinformatics | Probabilistic deep learning for single-cell multi-omics analysis including scVI, scANVI, totalVI, and spatial deconvolution. | |
| cellxgene-census-query | Data Query | Query CZ CELLxGENE Census (61M+ cells) for single-cell expression data by cell type, tissue, or disease. | |
| spatial-transcriptomics-spatial-data-io | Data I/O | Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, and other platforms using Squidpy and SpatialData. | |
| single-cell-atac-seq-qc-processing | ATAC-seq | Trim adapters, align reads, remove duplicates and mitochondrial contamination, and evaluate chromatin accessibility data quality. Includes TSS enrichment scoring and fragment size analysis. | |
| single-cell-atac-seq-peak-calling-annotaion | ATAC-seq | Call accessible chromatin peaks with MACS2, annotate peaks to genomic features and genes, and identify differentially accessible regions (DARs) between conditions. | |
| single-cell-proteomics-data-processing | Mass Spectrometry | Load, inspect, centroid, and extract features from raw LC-MS/MS data files using pyOpenMS. Includes TIC plotting, feature detection, and format conversion. | |
| single-cell-proteomics-peptide-identification | Mass Spectrometry | Search MS2 spectra against protein databases with MSFragger/Comet, apply target-decoy FDR filtering, and perform protein inference with parsimony principle. | |
| single-cell-multi-omics-data-harmonization | Data Integration | Prepare multi-omics datasets (RNA-seq, proteomics, methylation) for joint integration with per-assay normalization, batch correction, feature ID alignment, and missing value handling. | |

### Data Retrieval & Knowledge

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| pubchem-query | Chemical Database | Query PubChem database for chemical structures, similar compounds (similarity search), and bioactivity data against protein targets. | Recommended |
| uniprot-query | Protein Database | Query UniProt database for protein sequences, comprehensive metadata (function, domains, diseases), and search by gene name, organism, or keywords. | Recommended |
| chembl-query | Bioactivity Database | Query ChEMBL database for bioactivity data on drug-like compounds by target, molecule, or disease indication. | Recommended |
| kegg-query | Pathway Database | Query KEGG database for drug information, pathway analysis, and disease-drug-target discovery. | Recommended |
| ppi-string-query | PPI Database | Query STRING database for protein-protein interactions with confidence scores for network analysis. | Recommended |
| biomedical-literature-search | Literature Search | Search PubMed and bioRxiv for biomedical research papers with titles, abstracts, and metadata. | Recommended |

### Utilities

| Name | Category | Usage | Status |
|------|----------|-------|--------|
| biomed-skill-router | Skill Discovery | Find the most suitable skill for a given biomedical task by analyzing user requests and matching against available skill capabilities. | Recommended |
| biomed-skill-creator | Skill Development | Create new biomedical skills or improve existing ones through an interactive validation process with intent capture, workflow design, and evaluation. | Recommended |

---

## Notes

- **Recommended** skills contain `examples/`, `references/`, or `evals/` directories with supporting documentation and test cases.
- Skills without a status are still being validated and may require additional documentation and testing.
- Skills are organized by primary category for easy navigation.
- Each skill includes a SKILL.md file with detailed workflow documentation, usage examples, and error handling guidelines.

---

## Categories Summary

| Category | Count | Skills |
|----------|-------|--------|
| Biochemistry & Drug Discovery | 8 | drug-candidate-discovery, drug-lead-analysis, target-based-lead-design, admet-prediction, retrosynthesis-planning, iupac-name-identification-biot5, molecule-biochemical-significance-query-biot5, text-based-molecule-editing |
| Protein Analysis & Engineering | 12 | protein-mutation-analysis, mutation-design-aav, mutation-design-gfp, functional-protein-design, protein-function-prediction, similar-protein-retrieval, structure-prediction-boltz-2, protein-structure-design-boltzgen, antibody-structure-prediction-tfold, antibody-design-iggm, binding-affinity-prediction-prodigy, protein-ligand-binding-analysis-plip |
| Single-Cell Omics Data Analysis | 13 | single-cell-foundation-model-scrna-seq-geneformer, single-cell-foundation-model-scrna-seq-langcell, single-cell-foundation-model-scrna-seq-scgpt, spatial-transcriptomics-foundation-model-stofm, single-cell-scrna-seq-analysis-scanpy, single-cell-multi-omics-analysis-scvi, cellxgene-census-query, spatial-transcriptomics-spatial-data-io, single-cell-atac-seq-qc-processing, single-cell-atac-seq-peak-calling-annotaion, single-cell-proteomics-data-processing, single-cell-proteomics-peptide-identification, single-cell-multi-omics-data-harmonization |
| Data Retrieval & Knowledge | 6 | pubchem-query, uniprot-query, chembl-query, kegg-query, ppi-string-query, biomedical-literature-search |
| Utilities | 2 | biomed-skill-router, biomed-skill-creator |

---

## Quick Reference by Use Case

| I want to... | Use this skill |
|--------------|----------------|
| Design new drug candidates | `drug-candidate-discovery`, `target-based-lead-design` |
| Analyze molecule properties | `drug-lead-analysis`, `admet-prediction` |
| Find similar compounds | `pubchem-query` |
| Query protein database | `uniprot-query` |
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
| Process ATAC-seq data | `single-cell-atac-seq-qc-processing`, `single-cell-atac-seq-peak-calling-annotaion` |
| Process proteomics MS data | `single-cell-proteomics-data-processing`, `single-cell-proteomics-peptide-identification` |
| Harmonize multi-omics data | `single-cell-multi-omics-data-harmonization` |
| Search biomedical literature | `biomedical-literature-search` |
| Get IUPAC name of a molecule | `iupac-name-identification-biot5` |
| Understand molecule's biochemical significance | `molecule-biochemical-significance-query-biot5` |
| Plan retrosynthesis routes | `retrosynthesis-planning` |
| Find the right skill | `biomed-skill-router` |
| Create a new skill | `biomed-skill-creator` |
