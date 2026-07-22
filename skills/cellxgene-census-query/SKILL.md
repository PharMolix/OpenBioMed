---
name: cellxgene-census-query
description: >
  Query CZ CELLxGENE Census (61M+ cells). Filter by cell type/tissue/disease, retrieve expression data, and integrate with scanpy/PyTorch for population-scale single-cell analysis.
  Use this skill when:
  (1) Querying single-cell expression data by cell type, tissue, or disease,
  (2) Exploring available single-cell datasets and metadata,
  (3) Training machine learning models on single-cell data,
  (4) Performing large-scale cross-dataset analyses.
license: MIT
category: bioinformatics
tags: [single-cell, transcriptomics, genomics, scanpy, machine-learning, pytorch]
---

# CZ CELLxGENE Census

The CZ CELLxGENE Census provides programmatic access to a comprehensive, versioned collection of standardized single-cell genomics data from CZ CELLxGENE Discover. This skill enables efficient querying and analysis of millions of cells across thousands of datasets.

The Census includes:

- **61+ million cells** from human and mouse
- **Standardized metadata** (cell types, tissues, diseases, donors)
- **Raw gene expression** matrices
- **Pre-calculated embeddings** and statistics
- **Integration with PyTorch, scanpy, and other analysis tools**

## What it does

- Querying single-cell expression data by cell type, tissue, or disease
- Exploring available single-cell datasets and metadata
- Training machine learning models on single-cell data
- Performing large-scale cross-dataset analyses
- Integrating Census data with scanpy or other analysis frameworks
- Computing statistics across millions of cells
- Accessing pre-calculated embeddings or model predictions

## Why this exists

This skill encodes the correct, scalable methodological decisions for population-level single-cell data:
- Uses the official `tiledbsoma` backend to query data remotely *without* downloading massive files.
- Automatically handles out-of-core processing (`axis_query`) for datasets larger than your available RAM.
- Always enforces `is_primary_data == True` to prevent statistical inflation from duplicate cells.
- Native, memory-efficient integration directly into PyTorch DataLoaders and Scanpy objects.

---

## Handler API

Call the OpenBioMed API for CELLxGENE Census queries.

**Base URL**: `${OPENBIOMED_API_BASE_URL}` (resolved in order: env var → Docker default → default `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_pipeline/` | POST | Run census query operations |

### Operations

| Operation | Description | Required Parameters |
|-----------|-------------|---------------------|
| `get_summary` | Get census version and total cell counts | — |
| `get_datasets` | List available datasets with metadata | `organism` (optional) |
| `get_obs` | Query cell metadata by filters | `obs_value_filter` (optional) |
| `get_var` | Query gene metadata | `var_value_filter` (optional) |
| `get_anndata` | Retrieve expression data as AnnData | `obs_value_filter`, `var_value_filter` (optional) |

### Key Methodological Decisions (from original skill)

1. **Always use `is_primary_data == True`**: Prevents statistical inflation from duplicate cells — critical for population-level analysis
2. **Remote query without downloading**: Uses `tiledbsoma` backend to query data remotely — no need to download massive files
3. **Small-medium vs large queries**: 
   - `< 100k cells`: Use `get_anndata()` (fits in memory)
   - `> 100k cells`: Requires `axis_query()` for out-of-core processing (not in this tool)
4. **Specify `census_version`**: Use specific version for reproducible analyses (e.g., `"2023-07-25"`)
5. **Context manager pattern**: Always use `with` statement for automatic cleanup (handled internally)

### API Examples

#### Get Census Summary

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "cellxgene_census_query",
    "operation": "get_summary",
    "census_version": "stable"
  }'
```

**Response**:
```json
{
  "census_version": "stable",
  "total_cell_count": 61000000,
  "organism_counts": {"homo_sapiens": 50000000, "mus_musculus": 11000000},
  "message": "Census version stable contains 61,000,000 total cells"
}
```

#### List Available Datasets

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "cellxgene_census_query",
    "operation": "get_datasets",
    "organism": "homo_sapiens",
    "output_dir": "./tmp/"
  }'
```

**Response**:
```json
{
  "n_datasets": 1423,
  "csv_file": "./tmp/census_datasets_xxx.csv",
  "summary": {
    "n_datasets": 1423,
    "n_cells_total": 50000000,
    "unique_tissues": 56,
    "unique_diseases": 89
  },
  "message": "Found 1423 datasets for homo_sapiens. Saved to ./tmp/census_datasets_xxx.csv"
}
```

#### Query Cell Metadata

> **Important**: The tool automatically adds `is_primary_data == True` filter if not present.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "cellxgene_census_query",
    "operation": "get_obs",
    "organism": "homo_sapiens",
    "obs_value_filter": "cell_type == 'B cell' and tissue_general == 'lung'",
    "obs_column_names": ["cell_type", "tissue_general", "disease", "donor_id", "sex"]
  }'
```

**Response**:
```json
{
  "n_cells": 382194,
  "organism": "homo_sapiens",
  "obs_value_filter": "cell_type == 'B cell' and tissue_general == 'lung' and is_primary_data == True",
  "unique_counts": {"cell_type": 1, "tissue_general": 1, "disease": 14},
  "sample_values": {"cell_type": ["B cell"], "tissue_general": ["lung"]},
  "message": "Found 382,194 cells matching filter"
}
```

#### Query Gene Metadata

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "cellxgene_census_query",
    "operation": "get_var",
    "organism": "homo_sapiens",
    "var_value_filter": "feature_name in ['CD4', 'CD8A', 'CD19', 'FOXP3']"
  }'
```

**Response**:
```json
{
  "n_genes": 4,
  "organism": "homo_sapiens",
  "var_value_filter": "feature_name in ['CD4', 'CD8A', 'CD19', 'FOXP3']",
  "gene_names_sample": ["CD4", "CD8A", "CD19", "FOXP3"],
  "message": "Found 4 genes matching filter"
}
```

#### Retrieve Expression Data as AnnData

> **Note**: Maximum 100,000 cells by default. For larger queries, use more specific filters or implement out-of-core processing.

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "cellxgene_census_query",
    "operation": "get_anndata",
    "organism": "homo_sapiens",
    "obs_value_filter": "cell_type == 'neuron' and tissue_general == 'cortex'",
    "var_value_filter": "feature_name in ['FOXP2', 'TBR1', 'SATB2', 'NEUROD1']",
    "obs_column_names": ["cell_type", "tissue_general", "disease", "donor_id"],
    "output_dir": "./tmp/",
    "max_cells": 100000
  }'
```

**Response**:
```json
{
  "status": "success",
  "output_file": "./tmp/census_anndata_xxx.h5ad",
  "n_cells": 45678,
  "n_genes": 4,
  "organism": "homo_sapiens",
  "obs_value_filter": "cell_type == 'neuron' and tissue_general == 'cortex' and is_primary_data == True",
  "unique_counts": {"cell_type": 12, "tissue_general": 1, "disease": 5},
  "message": "Retrieved 45,678 cells x 4 genes. Saved to ./tmp/census_anndata_xxx.h5ad"
}
```

**If query exceeds max_cells**:
```json
{
  "status": "too_large",
  "n_cells_available": 250000,
  "max_cells": 100000,
  "suggestion": "Use more specific filters or axis_query for out-of-core processing",
  "message": "Query too large: 250,000 cells > 100,000 max. Please add more specific filters."
}
```

### Filter Syntax

- Use `and`, `or` for combining conditions
- Use `in` for multiple values: `tissue_general in ['lung', 'liver', 'brain']`
- Use `==` for exact match, `!=` for negation
- Common filters: `cell_type`, `tissue_general`, `disease`, `donor_id`, `sex`, `assay`

### Interpretation Guide

- **is_primary_data == True**: Essential filter — prevents counting the same cell multiple times from different dataset submissions
- **cell_type vs tissue_general**: `cell_type` is fine-grained (e.g., "CD4+ T cell"), `tissue_general` is broad (e.g., "lung")
- **Memory considerations**: AnnData with > 100k cells may exceed available RAM — use `axis_query()` for large-scale analyses
- **Census versioning**: Use specific version for reproducible analyses; `"stable"` points to latest stable release

## Usage

### 1. Opening the Census

Always use the context manager to ensure proper resource cleanup:

```python
import cellxgene_census

# Open latest stable version
with cellxgene_census.open_soma() as census:
    # Work with census data

# Open specific version for reproducibility
with cellxgene_census.open_soma(census_version="2023-07-25") as census:
    # Work with census data
```

**Key points:**

- Use context manager (`with` statement) for automatic cleanup
- Specify `census_version` for reproducible analyses
- Default opens latest "stable" release

### 2. Exploring Census Information

Before querying expression data, explore available datasets and metadata.

**Access summary information:**

```python
# Get summary statistics
summary = census["census_info"]["summary"].read().concat().to_pandas()
print(f"Total cells: {summary['total_cell_count'][0]}")

# Get all datasets
datasets = census["census_info"]["datasets"].read().concat().to_pandas()

# Filter datasets by criteria
covid_datasets = datasets[datasets["disease"].str.contains("COVID", na=False)]
```

**Query cell metadata to understand available data:**

```python
# Get unique cell types in a tissue
cell_metadata = cellxgene_census.get_obs(
    census,
    "homo_sapiens",
    value_filter="tissue_general == 'brain' and is_primary_data == True",
    column_names=["cell_type"]
)
unique_cell_types = cell_metadata["cell_type"].unique()
print(f"Found {len(unique_cell_types)} cell types in brain")

# Count cells by tissue
tissue_counts = cell_metadata.groupby("tissue_general").size()
```

**Important:** Always filter for `is_primary_data == True` to avoid counting duplicate cells unless specifically analyzing duplicates.

### 3. Querying Expression Data (Small to Medium Scale)

For queries returning < 100k cells that fit in memory, use `get_anndata()`:

```python
# Basic query with cell type and tissue filters
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",  # or "Mus musculus"
    obs_value_filter="cell_type == 'B cell' and tissue_general == 'lung' and is_primary_data == True",
    obs_column_names=["assay", "disease", "sex", "donor_id"],
)

# Query specific genes with multiple filters
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    var_value_filter="feature_name in ['CD4', 'CD8A', 'CD19', 'FOXP3']",
    obs_value_filter="cell_type == 'T cell' and disease == 'COVID-19' and is_primary_data == True",
    obs_column_names=["cell_type", "tissue_general", "donor_id"],
)
```

**Filter syntax:**

- Use `obs_value_filter` for cell filtering
- Use `var_value_filter` for gene filtering
- Combine conditions with `and`, `or`
- Use `in` for multiple values: `tissue in ['lung', 'liver']`
- Select only needed columns with `obs_column_names`

**Getting metadata separately:**

```python
# Query cell metadata
cell_metadata = cellxgene_census.get_obs(
    census, "homo_sapiens",
    value_filter="disease == 'COVID-19' and is_primary_data == True",
    column_names=["cell_type", "tissue_general", "donor_id"]
)

# Query gene metadata
gene_metadata = cellxgene_census.get_var(
    census, "homo_sapiens",
    value_filter="feature_name in ['CD4', 'CD8A']",
    column_names=["feature_id", "feature_name", "feature_length"]
)
```

### 4. Large-Scale Queries (Out-of-Core Processing)

For queries exceeding available RAM, use `axis_query()` with iterative processing:

```python
import tiledbsoma as soma

# Create axis query
query = census["census_data"]["homo_sapiens"].axis_query(
    measurement_name="RNA",
    obs_query=soma.AxisQuery(
        value_filter="tissue_general == 'brain' and is_primary_data == True"
    ),
    var_query=soma.AxisQuery(
        value_filter="feature_name in ['FOXP2', 'TBR1', 'SATB2']"
    )
)

# Iterate through expression matrix in chunks
iterator = query.X("raw").tables()
for batch in iterator:
    # batch is a pyarrow.Table with columns:
    # - soma_data: expression value
    # - soma_dim_0: cell (obs) coordinate
    # - soma_dim_1: gene (var) coordinate
    process_batch(batch)
```

**Computing incremental statistics:**

```python
# Example: Calculate mean expression
n_observations = 0
sum_values = 0.0

iterator = query.X("raw").tables()
for batch in iterator:
    values = batch["soma_data"].to_numpy()
    n_observations += len(values)
    sum_values += values.sum()

mean_expression = sum_values / n_observations
```

### 5. Machine Learning with PyTorch

For training models, use the experimental PyTorch integration:

```python
from cellxgene_census.experimental.ml import experiment_dataloader

with cellxgene_census.open_soma() as census:
    # Create dataloader
    dataloader = experiment_dataloader(
        census["census_data"]["homo_sapiens"],
        measurement_name="RNA",
        X_name="raw",
        obs_value_filter="tissue_general == 'liver' and is_primary_data == True",
        obs_column_names=["cell_type"],
        batch_size=128,
        shuffle=True,
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            X = batch["X"]  # Gene expression tensor
            labels = batch["obs"]["cell_type"]  # Cell type labels

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**Train/test splitting:**

```python
from cellxgene_census.experimental.ml import ExperimentDataset

# Create dataset from experiment
dataset = ExperimentDataset(
    experiment_axis_query,
    layer_name="raw",
    obs_column_names=["cell_type"],
    batch_size=128,
)

# Split into train and test
train_dataset, test_dataset = dataset.random_split(
    split=[0.8, 0.2],
    seed=42
)
```

### 6. Integration with Scanpy

Seamlessly integrate Census data with scanpy workflows:

```python
import scanpy as sc

# Load data from Census
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="cell_type == 'neuron' and tissue_general == 'cortex' and is_primary_data == True",
)

# Standard scanpy workflow
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Dimensionality reduction
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Visualization
sc.pl.umap(adata, color=["cell_type", "tissue", "disease"])
```

### 7. Multi-Dataset Integration

Query and integrate multiple datasets:

```python
# Strategy 1: Query multiple tissues separately
tissues = ["lung", "liver", "kidney"]
adatas = []

for tissue in tissues:
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=f"tissue_general == '{tissue}' and is_primary_data == True",
    )
    adata.obs["tissue"] = tissue
    adatas.append(adata)

# Concatenate
combined = adatas[0].concatenate(adatas[1:])

# Strategy 2: Query multiple datasets directly
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="tissue_general in ['lung', 'liver', 'kidney'] and is_primary_data == True",
)
```

### Use Case 1: Explore Cell Types in a Tissue

```python
with cellxgene_census.open_soma() as census:
    cells = cellxgene_census.get_obs(
        census, "homo_sapiens",
        value_filter="tissue_general == 'lung' and is_primary_data == True",
        column_names=["cell_type"]
    )
    print(cells["cell_type"].value_counts())
```

### Use Case 2: Query Marker Gene Expression

```python
with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        var_value_filter="feature_name in ['CD4', 'CD8A', 'CD19']",
        obs_value_filter="cell_type in ['T cell', 'B cell'] and is_primary_data == True",
    )
```

### Use Case 3: Train Cell Type Classifier

```python
from cellxgene_census.experimental.ml import experiment_dataloader

with cellxgene_census.open_soma() as census:
    dataloader = experiment_dataloader(
        census["census_data"]["homo_sapiens"],
        measurement_name="RNA",
        X_name="raw",
        obs_value_filter="is_primary_data == True",
        obs_column_names=["cell_type"],
        batch_size=128,
        shuffle=True,
    )

    # Train model
    for epoch in range(epochs):
        for batch in dataloader:
            # Training logic
            pass
```

### Use Case 4: Cross-Tissue Analysis

```python
with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter="cell_type == 'macrophage' and tissue_general in ['lung', 'liver', 'brain'] and is_primary_data == True",
    )

    # Analyze macrophage differences across tissues
    sc.tl.rank_genes_groups(adata, groupby="tissue_general")
```

## Example Output

```text
CZ CELLxGENE Census Query
==========================
Census Version: 2023-07-25 (Stable)
Organism: Homo sapiens
Filters: tissue_general == 'lung' AND is_primary_data == True

Query Summary:
  Found 382,194 unique cells across 14 datasets.
  Retrieved 2,000 highly variable genes.

Expression Data Loaded:
  AnnData object with n_obs × n_vars = 382194 × 2000
    obs: 'assay', 'cell_type', 'disease', 'tissue', 'donor_id'
    var: 'feature_id', 'feature_name', 'feature_length'

Downstream Ready:
  Memory footprint: ~3.1 GB
  Matrix format: scipy.sparse.csr_matrix

```

## Bundled Resources

This skill includes detailed reference documentation:
### references/census_schema.md

Comprehensive documentation of:

- Census data structure and organization
- All available metadata fields
- Value filter syntax and operators
- SOMA object types
- Data inclusion criteria

**When to read:** When you need detailed schema information, full list of metadata fields, or complex filter syntax.

### references/common_patterns.md

Examples and patterns for:

- Exploratory queries (metadata only)
- Small-to-medium queries (AnnData)
- Large queries (out-of-core processing)
- PyTorch integration
- Scanpy integration workflows
- Multi-dataset integration
- Best practices and common pitfalls

**When to read:** When implementing specific query patterns, looking for code examples, or troubleshooting common issues.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| cellxgene-census | latest |
| tiledbsoma | latest |
| scanpy | latest |
| pyarrow | latest |
| pandas | latest |
| numpy | latest |

### Inputs

| Name | Type | Format | Description |
|------|------|--------|-------------|
| query_parameters | parameters | string | Filters for cell type, tissue, disease, genes (e.g., obs_value_filter, var_value_filter) |

### Outputs

| Name | Type | Format | Description |
|------|------|--------|-------------|
| expression_data | object | anndata, h5ad | Single-cell expression matrices and metadata loaded into memory or saved to disk |
| ml_dataloader | object | pytorch-dataloader | Iterative dataloader for machine learning model training |

## Citations

https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills/blob/main/skills/cellxgene-census
