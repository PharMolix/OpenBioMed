---
name: single-cell-multi-omics-analysis-scvi
description: >
  Probabilistic deep learning framework for single-cell multi-omics data analysis.
  Use this skill when:
  (1) Analyzing single-cell RNA-seq data with batch correction,
  (2) Integrating multi-modal data (CITE-seq, ATAC-seq, multi-omics),
  (3) Performing cell type annotation with scANVI,
  (4) Spatial transcriptomics deconvolution with DestVI.
license: MIT
category: bioinformatics
tags: [single-cell, scRNA-seq, scATAC-seq, deep-learning, probabilistic-modeling, PyTorch]
---

# scvi-tools Single-Cell Multi-Omics Analysis

scvi-tools is a comprehensive Python framework for probabilistic models in single-cell genomics. Built on PyTorch and PyTorch Lightning, it provides deep generative models using variational inference for analyzing diverse single-cell data modalities.

## What it does

scvi-tools provides models organized by data modality:
### 1. Single-Cell RNA-seq Analysis

Core models for expression analysis, batch correction, and integration. See `references/models-scrna-seq.md` for:

- **scVI**: Unsupervised dimensionality reduction and batch correction
- **scANVI**: Semi-supervised cell type annotation and integration
- **AUTOZI**: Zero-inflation detection and modeling
- **VeloVI**: RNA velocity analysis
- **contrastiveVI**: Perturbation effect isolation
### 2. Chromatin Accessibility (ATAC-seq)

Models for analyzing single-cell chromatin data. See `references/models-atac-seq.md` for:

- **PeakVI**: Peak-based ATAC-seq analysis and integration
- **PoissonVI**: Quantitative fragment count modeling
- **scBasset**: Deep learning approach with motif analysis

### 3. Multimodal & Multi-omics Integration

Joint analysis of multiple data types. See `references/models-multimodal.md` for:

- **totalVI**: CITE-seq protein and RNA joint modeling
- **MultiVI**: Paired and unpaired multi-omic integration
- **MrVI**: Multi-resolution cross-sample analysis

### 4. Spatial Transcriptomics

Spatially-resolved transcriptomics analysis. See `references/models-spatial.md` for:

- **DestVI**: Multi-resolution spatial deconvolution
- **Stereoscope**: Cell type deconvolution
- **Tangram**: Spatial mapping and integration
- **scVIVA**: Cell-environment relationship analysis

### 5. Specialized Modalities

Additional specialized analysis tools. See `references/models-specialized.md` for:

- **MethylVI/MethylANVI**: Single-cell methylation analysis
- **CytoVI**: Flow/mass cytometry batch correction
- **Solo**: Doublet detection
- **CellAssign**: Marker-based cell type annotation
## Why this exists

This skill provides a unified, statistically rigorous foundation:
- **Raw counts directly**: No need for arbitrary pseudo-counts or log-normalization steps prior to modeling.
- **Unified API**: A consistent interface (setup → train → extract) across all models and multi-omic data types.
- **Principled Batch Correction**: Handles technical variation through explicit covariate registration in the generative model.
- **GPU Acceleration**: Automatically utilizes available GPUs via PyTorch Lightning to scale to millions of cells.

## Usage

All scvi-tools models follow a consistent API pattern:

```python
# 1. Load and preprocess data (AnnData format)
import scvi
import scanpy as sc

adata = scvi.data.heart_cell_atlas_subsampled()
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.highly_variable_genes(adata, n_top_genes=1200)

# 2. Register data with model (specify layers, covariates)
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",  # Use raw counts, not log-normalized
    batch_key="batch",
    categorical_covariate_keys=["donor"],
    continuous_covariate_keys=["percent_mito"]
)

# 3. Create and train model
model = scvi.model.SCVI(adata)
model.train()

# 4. Extract latent representations and normalized values
latent = model.get_latent_representation()
normalized = model.get_normalized_expression(library_size=1e4)

# 5. Store in AnnData for downstream analysis
adata.obsm["X_scVI"] = latent
adata.layers["scvi_normalized"] = normalized

# 6. Downstream analysis with scanpy
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

**Key Design Principles:**

- **Raw counts required**: Models expect unnormalized count data for optimal performance
- **Unified API**: Consistent interface across all models (setup → train → extract)
- **AnnData-centric**: Seamless integration with the scanpy ecosystem
- **GPU acceleration**: Automatic utilization of available GPUs
- **Batch correction**: Handle technical variation through covariate registration

### Common Analysis Tasks

#### Differential Expression

Probabilistic DE analysis using the learned generative models:

```python
de_results = model.differential_expression(
    groupby="cell_type",
    group1="TypeA",
    group2="TypeB",
    mode="change",  # Use composite hypothesis testing
    delta=0.25      # Minimum effect size threshold
)
```

See `references/differential-expression.md` for detailed methodology and interpretation.
#### Model Persistence

Save and load trained models:

```python
# Save model
model.save("./model_directory", overwrite=True)

# Load model
model = scvi.model.SCVI.load("./model_directory", adata=adata)
```

#### Batch Correction and Integration

Integrate datasets across batches or studies:

```python
# Register batch information
scvi.model.SCVI.setup_anndata(adata, batch_key="study")

# Model automatically learns batch-corrected representations
model = scvi.model.SCVI(adata)
model.train()
latent = model.get_latent_representation()  # Batch-corrected
```

## Example Output

```text
Training Status:
Epoch 1/400:   0%|          | 0/400 [00:00<?, ?it/s]
Epoch 400/400: 100%|██████████| 400/400 [02:15<00:00,  2.95it/s, v_num=1]
Training finished.
Final ELBO loss: 1245.67

Outputs generated:
  1. Model Directory: ./model_directory/
     - model.pt (Trained neural network weights)
     - attr.pkl (Model hyperparameters and architecture)
     - var_names.csv (Features used for training)

  2. Updated AnnData object (latent_anndata.h5ad):
     AnnData object with n_obs × n_vars = 15000 × 1200
       obs: 'batch', 'donor', 'cell_type', '_scvi_batch', '_scvi_labels'
       var: 'highly_variable', 'means', 'variances'
       uns: '_scvi_uuid', '_scvi_manager_uuid'
       obsm: 'X_scVI' (10-dimensional batch-corrected latent space)
       layers: 'counts', 'scvi_normalized' (Denoised expected expression)
```

## Best practice

1. **Use raw counts**: Always provide unnormalized count data to models
2. **Filter genes**: Remove low-count genes before analysis (e.g., `min_counts=3`)
3. **Register covariates**: Include known technical factors (batch, donor, etc.) in `setup_anndata`
4. **Feature selection**: Use highly variable genes for improved performance
5. **Model saving**: Always save trained models to avoid retraining
6. **GPU usage**: Enable GPU acceleration for large datasets (`accelerator="gpu"`)
7. **Scanpy integration**: Store outputs in AnnData objects for downstream analysis

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| scvi-tools | latest |
| scanpy | latest |
| anndata | latest |
| torch | latest |
| pytorch-lightning | latest |
| cuda | Recommended for GPU acceleration |

### Inputs

| Name | Type | Format | Description |
|------|------|--------|-------------|
| anndata | file | h5ad | AnnData object containing raw, unnormalized count data |

### Outputs

| Name | Type | Format | Description |
|------|------|--------|-------------|
| latent_anndata | file | h5ad | Updated AnnData object containing batch-corrected latent representations and normalized values |
| model_dir | directory | pt, pkl | Saved scvi-tools model directory for future inference |

## Citations

https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills/blob/main/skills/scvi-tools
