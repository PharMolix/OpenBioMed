---
name: spatial-transcriptomics-spatial-data-io
description: >
  Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, and other platforms using Squidpy and SpatialData.
  Use this skill when:
  (1) Loading Visium spatial transcriptomics data from Space Ranger output,
  (2) Loading Xenium single-cell resolution spatial data,
  (3) Loading MERFISH, CosMx, or other spatial platforms,
  (4) Converting between SpatialData and AnnData formats.
license: MIT
category: bioinformatics
tags: [spatial-transcriptomics, io, squidpy, spatialdata, visium, xenium, merfish]
---

# Spatial Transcriptomics Data I/O

Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, and other platforms via OpenBioMed API.

## When to Use

- User provides spatial transcriptomics data directory and wants to load it
- User asks to load Visium, Xenium, MERFISH, or other spatial data formats
- User wants to convert between AnnData and SpatialData formats
- User has platform-specific output directory (Space Ranger, Xenium, MERSCOPE, etc.)

## API Endpoint

**Base URL**: `${OPENBIOMED_API_BASE_URL}` (resolved in order: env var → Docker default → local `http://127.0.0.1:8095`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_pipeline/` | POST | Load spatial transcriptomics data |

## Workflow

### Step 1: Prepare Data Directory

Ensure your spatial transcriptomics data is accessible on the server. The data directory must contain platform-specific files.

| Platform | Required Files |
|----------|----------------|
| **Visium** | `filtered_feature_bc_matrix.h5`, `tissue_positions_list.csv`, tissue image |
| **Xenium** | `cells_summary.parquet`, `cell_features.parquet` |
| **MERSCOPE** | `cell_by_gene.csv`, `cell_metadata.csv` |
| **Slide-seq** | Beads CSV file, coordinates CSV file |
| **CosMx** | Platform-specific output files |
| **Stereo-seq** | Platform-specific output files |

### Step 2: Call API to Load Data

Submit a request to load the spatial data:

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "spatial_transcriptomics_loading",
    "value": "/path/to/spaceranger/output",
    "query": "visium",
    "mode": "anndata"
  }'
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | Yes | Must be `"spatial_transcriptomics_loading"` |
| `value` | string | Yes | Path to data directory on server |
| `query` | string | No | Platform type (default: `"visium"`). Options: `visium`, `xenium`, `merscope`, `slideseq`, `cosmx`, `stereoseq` |
| `mode` | string | No | Output format (default: `"anndata"`). Options: `anndata`, `spatialdata` |
| `dataset` | string | No | Library ID for Visium data (optional) |

### Step 3: Receive Loaded Data

The API returns metadata about the loaded data:

```json
{
  "task": "spatial_transcriptomics_loading",
  "data_file": "./tmp/spatial_data_xxxx.h5ad",
  "platform": "visium",
  "n_obs": 2987,
  "n_vars": 31053,
  "has_spatial_coords": true,
  "has_images": true,
  "output_format": "anndata",
  "description": "Loaded Visium data with 2987 spots and 31053 genes. Saved to ./tmp/spatial_data_xxxx.h5ad"
}
```

**Output Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `data_file` | string | Path to saved .h5ad or .zarr file |
| `platform` | string | Platform type used |
| `n_obs` | int | Number of spots/cells loaded |
| `n_vars` | int | Number of genes |
| `has_spatial_coords` | bool | Whether spatial coordinates are available |
| `has_images` | bool | Whether tissue images are loaded |
| `output_format` | string | Output format used |
| `description` | string | Summary message |

## Example Usage

### Example 1: Load Visium Data

```
Input: "Load my Visium data from Space Ranger output at /data/sample1/"

Step 1: Data directory contains Space Ranger output files
Step 2: Call API

curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "spatial_transcriptomics_loading",
    "value": "/data/sample1",
    "query": "visium",
    "mode": "anndata"
  }'

Response:
{
  "task": "spatial_transcriptomics_loading",
  "data_file": "./tmp/spatial_data_abc123.h5ad",
  "platform": "visium",
  "n_obs": 2987,
  "n_vars": 31053,
  "has_spatial_coords": true,
  "has_images": true,
  "output_format": "anndata",
  "description": "Loaded Visium data with 2987 spots and 31053 genes"
}

Output: AnnData object saved to ./tmp/spatial_data_abc123.h5ad
Ready for downstream analysis with Squidpy/Scanpy
```

### Example 2: Load Xenium Single-Cell Data

```
Input: "Load Xenium data from /data/xenium_run/"

Step 1: Data directory contains Xenium output files
Step 2: Call API

curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "spatial_transcriptomics_loading",
    "value": "/data/xenium_run",
    "query": "xenium",
    "mode": "anndata"
  }'

Response:
{
  "task": "spatial_transcriptomics_loading",
  "data_file": "./tmp/spatial_data_def456.h5ad",
  "platform": "xenium",
  "n_obs": 50000,
  "n_vars": 500,
  "has_spatial_coords": true,
  "has_images": false,
  "output_format": "anndata",
  "description": "Loaded Xenium data with 50000 cells and 500 genes"
}

Output: Single-cell resolution spatial data loaded
```

### Example 3: Load MERSCOPE Data as SpatialData

```
Input: "Load MERSCOPE data from /data/merscope_exp/ in SpatialData format"

Step 1: Data directory contains MERSCOPE output
Step 2: Call API with spatialdata format

curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "spatial_transcriptomics_loading",
    "value": "/data/merscope_exp",
    "query": "merscope",
    "mode": "spatialdata"
  }'

Response:
{
  "task": "spatial_transcriptomics_loading",
  "data_file": "./tmp/spatial_data_ghi789.zarr",
  "platform": "merscope",
  "n_obs": 25000,
  "n_vars": 500,
  "has_spatial_coords": true,
  "has_images": true,
  "output_format": "spatialdata",
  "description": "Loaded MERSCOPE data with 25000 cells"
}

Output: SpatialData object saved as Zarr format
```

## Supported Platforms

| Platform | Description | Typical Resolution |
|----------|-------------|--------------------|
| `visium` | 10x Genomics Visium | Spot-level (~55μm spots) |
| `xenium` | 10x Genomics Xenium | Single-cell resolution |
| `merscope` | Vizgen MERFISH/MERSCOPE | Single-cell resolution |
| `slideseq` | Slide-seq / Slide-seqV2 | Near single-cell |
| `cosmx` | Nanostring CosMx | Single-cell/subcellular |
| `stereoseq` | BGI Stereo-seq | Single-cell/near single-cell |

## Expected Outputs

| Format | File Extension | Description |
|--------|----------------|-------------|
| `anndata` | `.h5ad` | AnnData object for Scanpy/Squidpy |
| `spatialdata` | `.zarr` | SpatialData object for multi-modal spatial |

The loaded data contains:
- **Expression matrix**: Gene counts per spot/cell
- **Spatial coordinates**: X/Y positions in `obsm['spatial']`
- **Images**: Tissue images in `uns['spatial']` (when available)
- **Scale factors**: For coordinate-to-image mapping

## Error Handling

### Data Directory Not Found

**Symptom**: API returns 500 error with "Data directory not found".

**Solution**: Verify the path exists on the server:
```bash
ls /path/to/data_dir
```

### Platform Files Missing

**Symptom**: API returns error "Visium data not found. Expected filtered_feature_bc_matrix.h5".

**Solution**: Check for required files in the output directory. For Visium, Space Ranger output typically has files in `outs/` subdirectory.

### Unsupported Platform

**Symptom**: API returns "Unsupported platform: xxx".

**Solution**: Use one of supported platforms: `visium`, `xenium`, `merscope`, `slideseq`, `cosmx`, `stereoseq`.

### SpatialData Format Not Available

**Symptom**: API falls back to AnnData format even when `mode: spatialdata` requested.

**Solution**: Ensure `spatialdata` and `spatialdata-io` packages are installed on the server. AnnData format works as fallback.

## Dependencies

| Package | Version | Required For |
|---------|---------|--------------|
| squidpy | 1.3+ | Visium, Xenium, MERSCOPE, Slide-seq |
| spatialdata | 0.1+ | SpatialData output format |
| spatialdata_io | latest | CosMx, Stereo-seq loading |
| scanpy | 1.10+ | AnnData operations |
| anndata | 0.10+ | All formats |

## Interpretation Guide

### Spot vs Cell Resolution

| Platform | Resolution | Analysis Considerations |
|----------|------------|------------------------|
| Visium | Spots (~55μm) | May contain multiple cells; use deconvolution methods |
| Xenium, MERSCOPE | Single-cell | Direct cell-level analysis possible |
| Slide-seq | Near single-cell | Small beads, but may still mix cells |
| CosMx, Stereo-seq | Single/subcellular | High resolution, subcellular features available |

### Coordinate Systems

- **Visium**: Pixel coordinates relative to tissue image
- **Xenium/MERSCOPE**: Physical coordinates in micrometers
- Always check `obsm['spatial']` units before downstream analysis

## See Also

- `single-cell-scrna-seq-analysis-scanpy` - Scanpy analysis pipeline
- `spatial-transcriptomics-foundation-model-stofm` - STofM spatial foundation model
- `single-cell-multi-omics-analysis-scvi` - Multi-modal analysis with scVI

## Citations

- Squidpy: Sturmhöfel et al., Nature Methods 2022
- SpatialData: Marconato et al., Nature Methods 2024