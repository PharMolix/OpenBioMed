---
name: single-cell-scrna-seq-analysis-scanpy
description: >
  Complete single-cell RNA-seq analysis workflow built on Scanpy and AnnData.
  Use this skill when:
  (1) Loading diverse single-cell data formats (10X, h5ad, CSV),
  (2) Performing quality control and filtering,
  (3) Normalization, dimensionality reduction, and clustering,
  (4) Marker gene identification and cell type annotation.
license: MIT
category: bioinformatics
tags: [single-cell, RNA-seq, scanpy, transcriptomics, dimensionality-reduction, clustering]
---

# Scanpy Single-Cell Analysis

Scanpy is a scalable Python toolkit for analyzing single-cell RNA-seq data, built on AnnData. Apply this skill for complete single-cell workflows including quality control, normalization, dimensionality reduction, clustering, marker gene identification, visualization, and trajectory analysis.

## What it does

1. **Loads diverse data formats:** Ingests 10X Genomics (`.h5`, `.mtx`), `.h5ad`, or `.csv` files into the AnnData structure.
2. **Automates Quality Control (QC):** Identifies mitochondrial genes, calculates QC metrics, and filters low-quality cells/genes.
3. **Preprocesses and Normalizes:** Performs log-transformation, total-count normalization (typically 10k/cell), and highly variable gene (HVG) selection, safely backing up raw counts.
4. **Reduces Dimensionality:** Computes PCA and builds neighborhood graphs, followed by UMAP or t-SNE embeddings.
5. **Clusters Cells:** Applies Leiden community detection across multiple resolutions to find optimal granularities.
6. **Identifies Marker Genes:** Uses statistical testing (Wilcoxon) to rank genes driving cluster identity, enabling manual or automated cell type annotation.
7. **Generates Publication-Ready Plots:** Produces highly customized, high-DPI violin plots, UMAPs, heatmaps, and dot plots.

## Why this exists

This skill encodes the current best practices for scRNA-seq:
- Enforces strict tracking of metadata (`adata.obs` and `adata.var`).
- Utilizes the more robust Leiden algorithm over Louvain.
- Bundles automated, reproducible scripts for both QC and full-pipeline analysis.
- Provides immediate access to trajectory inference (PAGA/DPT) and gene set scoring.


---

## Handler API

Call the OpenBioMed API for single-cell RNA-seq analysis using Scanpy.

**Base URL**: `${OPENBIOMED_API_BASE_URL}` (resolved in order: env var → Docker default → default `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_pipeline/` | POST | Run analysis operations |
| `/api/upload` | POST | Upload user files to server |

## Input File Handling

| Input Type | How to Handle |
|------------|---------------|
| **Uploaded file** | Use file_id with `/api/upload` endpoint first |
| Server directory | Use path directly if data exists on server |
| Compressed archive | Upload and extract on server |

### Uploading User Files

When the user has uploaded a single-cell data file (e.g., h5ad, h5, mtx, csv), you will see a file_id (UUID format) in the conversation. Use the `http_request` tool to upload it to the server:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/api/upload" \
  -F "file=@<file_path>"
```

Or using http_request tool:
```
url: "${OPENBIOMED_API_BASE_URL}/api/upload"
method: "POST"
files: '{"file": "<file_id>"}'
```

The system will automatically:
- Resolve the file_id to the actual file on disk
- Read the file bytes and send as multipart/form-data
- Inject the required API Key header

**Response**: `{"path": "./tmp/uploads/<uuid>.h5ad"}`

Use this `path` value as the `file_path` parameter in subsequent API calls.

### If Input is Server Directory

If single-cell data already exists on server:

```bash
# Verify directory exists
ls /path/to/data.h5ad
```

Use the path directly as `file_path` parameter.

### Operations

| Operation | Description | Required Parameters |
|-----------|-------------|---------------------|
| `load` | Load h5ad, h5, mtx, or CSV files | `file_path` |
| `qc` | Quality control and filtering | `file_path`, `min_genes`, `min_cells`, `max_mt_percent` |
| `normalize` | Normalization, log-transformation, HVG selection | `file_path`, `n_top_genes` |
| `cluster` | PCA, UMAP, and Leiden clustering | `file_path`, `n_neighbors`, `n_pcs`, `resolution` |
| `markers` | Marker gene identification (Wilcoxon) | `file_path`, `groupby` |
| `full_pipeline` | Complete workflow (load → qc → normalize → cluster → markers) | `file_path` |

### Key Methodological Decisions (from original skill)

1. **QC before analysis**: Identify mitochondrial genes (`MT-` for human, `mt-` for mouse), calculate QC metrics, filter low-quality cells/genes
2. **Normalization strategy**: Total-count normalization (10,000 counts/cell) + log-transformation — standard for scRNA-seq
3. **Save raw counts backup**: Store `adata.raw = adata` before normalization for later use
4. **Leiden clustering**: Use Leiden (more robust than Louvain) for community detection
5. **HVG selection**: Select highly variable genes (typically 2000) for dimensionality reduction
6. **Wilcoxon test for markers**: Statistical testing (Wilcoxon rank-sum) for marker gene identification

### API Examples

#### Example 0: Upload and Analyze User's h5ad File

```
Input: "I've uploaded my single-cell data file (h5ad). Please run QC and clustering."

Step 1: Upload file to server
User has uploaded file with file_id: "550e8400-e29b-41d4-a716-446655440000"

Upload to server:
curl -X POST "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/api/upload" \
  -F "file=@/path/to/uploaded_file.h5ad"

Response: {"path": "./tmp/uploads/550e8400.h5ad"}

Step 2: Run full pipeline on uploaded file
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "full_pipeline",
    "file_path": "./tmp/uploads/550e8400.h5ad",
    "min_genes": 200,
    "min_cells": 3,
    "max_mt_percent": 5.0,
    "n_top_genes": 2000,
    "n_neighbors": 10,
    "n_pcs": 40,
    "resolution": 0.5
  }'

Output: Full analysis pipeline executed on user's uploaded file
```

#### Load Data

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "load",
    "file_path": "path/to/data.h5ad",
    "output_dir": "./tmp/"
  }'
```

**Response**:
```json
{
  "output_file": "./tmp/loaded_xxx.h5ad",
  "format": "h5ad",
  "n_obs": 12450,
  "n_vars": 32000,
  "has_raw": false,
  "has_pca": false,
  "has_umap": false,
  "has_clusters": false,
  "message": "Loaded h5ad data: 12450 cells x 32000 genes. Saved to ./tmp/loaded_xxx.h5ad"
}
```

#### Quality Control

The `qc` operation automatically:
- Identifies mitochondrial genes (`MT-` or `mt-` prefix)
- Calculates QC metrics: `n_genes_by_counts`, `total_counts`, `pct_counts_mt`
- Filters cells by minimum genes
- Filters genes by minimum cells
- Removes high MT% cells (default: < 5%)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "qc",
    "file_path": "path/to/data.h5ad",
    "output_dir": "./tmp/",
    "min_genes": 200,
    "min_cells": 3,
    "max_mt_percent": 5.0
  }'
```

**Response**:
```json
{
  "output_file": "./tmp/filtered_xxx.h5ad",
  "pre_filter_cells": 12450,
  "pre_filter_genes": 32000,
  "post_filter_cells": 10210,
  "post_filter_genes": 18500,
  "cells_removed": 2240,
  "genes_removed": 13500,
  "min_genes": 200,
  "min_cells": 3,
  "max_mt_percent": 5.0,
  "mt_genes_detected": 13,
  "qc_plot": "./tmp/figures/qc_metrics_xxx.pdf",
  "message": "QC completed. Filtered 2240 cells, 13500 genes. Saved to ./tmp/filtered_xxx.h5ad"
}
```

#### Normalize and Select HVGs

The `normalize` operation:
- Total-count normalization (target_sum=10,000)
- Log-transformation (`log1p`)
- Saves raw counts backup (`adata.raw`)
- Identifies highly variable genes

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "normalize",
    "file_path": "path/to/filtered.h5ad",
    "output_dir": "./tmp/",
    "n_top_genes": 2000
  }'
```

**Response**:
```json
{
  "output_file": "./tmp/normalized_xxx.h5ad",
  "n_top_genes": 2000,
  "n_hvgs_detected": 2000,
  "mean_hvg_dispersion": 0.85,
  "target_sum": 10000,
  "hvg_plot": "./tmp/figures/hvg_xxx.pdf",
  "message": "Normalization completed. 2000 HVGs selected. Saved to ./tmp/normalized_xxx.h5ad"
}
```

#### Clustering (PCA + UMAP + Leiden)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "cluster",
    "file_path": "path/to/normalized.h5ad",
    "output_dir": "./tmp/",
    "n_neighbors": 10,
    "n_pcs": 40,
    "resolution": 0.5
  }'
```

**Response**:
```json
{
  "output_file": "./tmp/clustered_xxx.h5ad",
  "n_clusters": 12,
  "cluster_sizes": {"0": 3210, "1": 2800, "2": 2150, ...},
  "n_neighbors": 10,
  "n_pcs": 40,
  "resolution": 0.5,
  "umap_plot": "./tmp/figures/umap_clusters_xxx.pdf",
  "pca_plot": "./tmp/figures/pca_variance_xxx.pdf",
  "message": "Clustering completed. Found 12 clusters. Saved to ./tmp/clustered_xxx.h5ad"
}
```

#### Marker Gene Identification

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "markers",
    "file_path": "path/to/clustered.h5ad",
    "output_dir": "./tmp/",
    "groupby": "leiden"
  }'
```

**Response**:
```json
{
  "output_file": "./tmp/with_markers_xxx.h5ad",
  "markers_csv": "./tmp/markers_xxx.csv",
  "n_groups": 12,
  "groupby": "leiden",
  "method": "wilcoxon",
  "top_markers": {
    "0": {"genes": ["CD3D", "IL7R", "CCR7"], "scores": [45.2, 38.1, 32.5]},
    "1": {"genes": ["CD14", "LYZ", "S100A9"], "scores": [52.3, 48.2, 41.7]},
    ...
  },
  "heatmap_plot": "./tmp/figures/markers_heatmap_xxx.pdf",
  "dotplot_plot": "./tmp/figures/markers_dotplot_xxx.pdf",
  "message": "Marker genes identified for 12 groups. Saved to ./tmp/markers_xxx.csv"
}
```

#### Run Full Pipeline

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "scanpy_analysis",
    "operation": "full_pipeline",
    "file_path": "path/to/data.h5ad",
    "output_dir": "./tmp/",
    "min_genes": 200,
    "min_cells": 3,
    "max_mt_percent": 5.0,
    "n_top_genes": 2000,
    "n_neighbors": 10,
    "n_pcs": 40,
    "resolution": 0.5
  }'
```

**Response**:
```json
{
  "output_file": "./tmp/pipeline_complete_xxx.h5ad",
  "pipeline_steps": ["load", "qc", "normalize", "cluster", "markers"],
  "load_stats": {"n_obs_initial": 12450, "n_vars_initial": 32000},
  "qc_stats": {"cells_removed": 2240, "genes_removed": 13500, "post_filter_cells": 10210},
  "normalize_stats": {"n_hvgs": 2000},
  "cluster_stats": {"n_clusters": 12, "resolution": 0.5},
  "markers_stats": {"n_groups": 12, "markers_csv": "./tmp/markers_xxx.csv"},
  "figures": {
    "qc_plot": "./tmp/figures/qc_metrics_xxx.pdf",
    "hvg_plot": "./tmp/figures/hvg_xxx.pdf",
    "pca_plot": "./tmp/figures/pca_variance_xxx.pdf",
    "umap_plot": "./tmp/figures/umap_clusters_xxx.pdf",
    "heatmap_plot": "./tmp/figures/markers_heatmap_xxx.pdf",
    "dotplot_plot": "./tmp/figures/markers_dotplot_xxx.pdf"
  },
  "message": "Full pipeline completed. 12 clusters found. Final data saved to ./tmp/pipeline_complete_xxx.h5ad"
}
```

### QC Thresholds Guide

| Parameter | Default | Typical Range | Meaning |
|-----------|---------|---------------|---------|
| `min_genes` | 200 | 200–500 | Cells with fewer genes are likely empty droplets |
| `min_cells` | 3 | 3–10 | Genes in fewer cells are likely noise |
| `max_mt_percent` | 5.0 | 5–20 | High MT% indicates dead/damaged cells |
| `n_top_genes` | 2000 | 2000–5000 | HVGs for dimensionality reduction |
| `resolution` | 0.5 | 0.3–1.0 | Higher = more clusters (granular) |

### Interpretation Guide

- **High MT% cells**: Cells with `pct_counts_mt > 20%` likely dead/damaged — filter them
- **Low gene count cells**: `n_genes_by_counts < 200` likely empty droplets or low-quality cells
- **Cluster resolution**: Try multiple resolutions (0.3, 0.5, 0.8) to find optimal granularity
- **Marker genes**: Check if top markers match expected cell types — validates clustering
- **HVG dispersion**: Higher mean dispersion indicates better separation between cell states

## Usage

### Quick Start

#### Basic Import and Setup

```python
import scanpy as sc
import pandas as pd
import numpy as np

# Configure settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.settings.figdir = './figures/'
```

#### Loading Data

```python
# From 10X Genomics
adata = sc.read_10x_mtx('path/to/data/')
adata = sc.read_10x_h5('path/to/data.h5')

# From h5ad (AnnData format)
adata = sc.read_h5ad('path/to/data.h5ad')

# From CSV
adata = sc.read_csv('path/to/data.csv')
```

#### Understanding AnnData Structure

The AnnData object is the core data structure in scanpy:

```python
adata.X          # Expression matrix (cells × genes)
adata.obs        # Cell metadata (DataFrame)
adata.var        # Gene metadata (DataFrame)
adata.uns        # Unstructured annotations (dict)
adata.obsm       # Multi-dimensional cell data (PCA, UMAP)
adata.raw        # Raw data backup

# Access cell and gene names
adata.obs_names  # Cell barcodes
adata.var_names  # Gene names
```

### Standard Analysis Workflow

#### 1. Quality Control

Identify and filter low-quality cells and genes:

```python
# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

# Visualize QC metrics
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

# Filter cells and genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 5, :]  # Remove high MT% cells
```

**Use the QC script for automated analysis:**

```shell
python scripts/qc_analysis.py input_file.h5ad --output filtered.h5ad
```

#### 2. Normalization and Preprocessing

```python
# Normalize to 10,000 counts per cell
sc.pp.normalize_total(adata, target_sum=1e4)

# Log-transform
sc.pp.log1p(adata)

# Save raw counts for later
adata.raw = adata

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pl.highly_variable_genes(adata)

# Subset to highly variable genes
adata = adata[:, adata.var.highly_variable]

# Regress out unwanted variation
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# Scale data
sc.pp.scale(adata, max_value=10)
```

#### 3. Dimensionality Reduction

```python
# PCA
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True)  # Check elbow plot

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# UMAP for visualization
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden')

# Alternative: t-SNE
sc.tl.tsne(adata)
```

#### 4. Clustering

```python
# Leiden clustering (recommended)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color='leiden', legend_loc='on data')

# Try multiple resolutions to find optimal granularity
for res in [0.3, 0.5, 0.8, 1.0]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
```

#### 5. Marker Gene Identification

```python
# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Visualize results
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)

# Get results as DataFrame
markers = sc.get.rank_genes_groups_df(adata, group='0')
```

#### 6. Cell Type Annotation

```python
# Define marker genes for known cell types
marker_genes = ['CD3D', 'CD14', 'MS4A1', 'NKG7', 'FCGR3A']

# Visualize markers
sc.pl.umap(adata, color=marker_genes, use_raw=True)
sc.pl.dotplot(adata, var_names=marker_genes, groupby='leiden')

# Manual annotation
cluster_to_celltype = {
    '0': 'CD4 T cells',
    '1': 'CD14+ Monocytes',
    '2': 'B cells',
    '3': 'CD8 T cells',
}
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_to_celltype)

# Visualize annotated types
sc.pl.umap(adata, color='cell_type', legend_loc='on data')
```

#### 7. Save Results

```python
# Save processed data
adata.write('results/processed_data.h5ad')

# Export metadata
adata.obs.to_csv('results/cell_metadata.csv')
adata.var.to_csv('results/gene_metadata.csv')
```

### Common Tasks

#### Creating Publication-Quality Plots

```python
# Set high-quality defaults
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(5, 5))
sc.settings.file_format_figs = 'pdf'

# UMAP with custom styling
sc.pl.umap(adata, color='cell_type',
           palette='Set2',
           legend_loc='on data',
           legend_fontsize=12,
           legend_fontoutline=2,
           frameon=False,
           save='_publication.pdf')

# Heatmap of marker genes
sc.pl.heatmap(adata, var_names=genes, groupby='cell_type',
              swap_axes=True, show_gene_labels=True,
              save='_markers.pdf')

# Dot plot
sc.pl.dotplot(adata, var_names=genes, groupby='cell_type',
              save='_dotplot.pdf')
```

Refer to `references/plotting_guide.md` for comprehensive visualization examples.

#### Trajectory Inference

```python
# PAGA (Partition-based graph abstraction)
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, color='leiden')

# Diffusion pseudotime
adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden'] == '0')[0]
sc.tl.dpt(adata)
sc.pl.umap(adata, color='dpt_pseudotime')
```

#### Differential Expression Between Conditions

```python
# Compare treated vs control within cell types
adata_subset = adata[adata.obs['cell_type'] == 'T cells']
sc.tl.rank_genes_groups(adata_subset, groupby='condition',
                         groups=['treated'], reference='control')
sc.pl.rank_genes_groups(adata_subset, groups=['treated'])
```

#### Gene Set Scoring

```python
# Score cells for gene set expression
gene_set = ['CD3D', 'CD3E', 'CD3G']
sc.tl.score_genes(adata, gene_set, score_name='T_cell_score')
sc.pl.umap(adata, color='T_cell_score')
```

#### Batch Correction

```python
# ComBat batch correction
sc.pp.combat(adata, key='batch')

# Alternative: use Harmony or scVI (separate packages)
```

## Example Output

```
Scanpy Single-Cell QC & Clustering
==================================
Input: 12,450 cells × 32,000 genes
After filtering (min_genes=200, MT% < 5): 10,210 cells × 18,500 genes

Variance explained by top 40 PCs: 68.4%

Clustering (Leiden, res=0.5):
  Found 12 distinct clusters.
  Cluster 0: 3,210 cells (Top markers: CD3D, IL7R) -> Annotated: CD4 T cells
  Cluster 1: 2,800 cells (Top markers: CD14, LYZ)  -> Annotated: CD14+ Monocytes

Saved outputs to: results/
  - processed_data.h5ad
  - UMAP_clusters.pdf (300 dpi)
  - marker_genes_dotplot.pdf (300 dpi)
```

## Best practice

1. **Start with the template**: Use `assets/analysis_template.py` as a starting point
2. **Run QC script first**: Use `scripts/qc_analysis.py` for initial filtering
3. **Consult references as needed**: Load workflow and API references into context
4. **Iterate on clustering**: Try multiple resolutions and visualization methods
5. **Validate biologically**: Check marker genes match expected cell types
6. **Document parameters**: Record QC thresholds and analysis settings
7. **Save checkpoints**: Write intermediate results at key steps

## Bundled Resources

This skill includes pre-configured tools to accelerate your workflow:

- **`scripts/qc_analysis.py`**: Automated QC script calculating metrics, generating plots, and filtering data.
- **`assets/analysis_template.py`**: A complete, customizable python template from data loading to cell type annotation.
- **`references/standard_workflow.md`**: Step-by-step conceptual explanations.
- **`references/api_reference.md`**: Quick lookup for Scanpy function signatures.
- **`references/plotting_guide.md`**: Customization guide for publication-ready figures.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| scanpy | latest |
| anndata | latest |
| pandas | latest |
| numpy | latest |
| matplotlib | latest |
| scikit-learn | latest |

### Inputs

| Name | Type | Format | Description |
|------|------|--------|-------------|
| count-matrix | file | h5ad, h5, mtx, csv | Raw or filtered gene expression counts (10X Genomics or AnnData formats) |

### Outputs

| Name | Type | Format | Description |
|------|------|--------|-------------|
| processed-anndata | file | h5ad | Fully processed AnnData object containing normalized counts, PCA, UMAP, and clustering |
| figures | file | png, pdf | Publication-quality UMAPs, marker gene heatmaps, and QC violin plots |
| metadata | file | csv | Exported cell and gene metadata tables |

## Citations

https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills/blob/main/skills/scanpy
