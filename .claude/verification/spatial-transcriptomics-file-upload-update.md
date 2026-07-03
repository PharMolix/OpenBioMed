---
name: spatial-transcriptomics-spatial-data-io-file-upload-update
description: Update spatial transcriptomics skill to support user uploaded h5ad files
---

# Modification Summary: h5ad File Upload Support

**Date**: 2026-07-03
**Status**: ✅ COMPLETED

## Overview

Added support for user uploaded h5ad files in spatial-transcriptomics-spatial-data-io skill.

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `skills/spatial-transcriptomics-spatial-data-io/SKILL.md` | +94 | Added file upload documentation and h5ad platform |
| `open_biomed/tools/spatial_transcriptomics_tool.py` | +40 | Added `_load_h5ad()` method and auto-detection |
| `open_biomed/scripts/run_server.py` | +5 | Updated handler docstring |

## Key Changes

### 1. SKILL.md Updates

**Added sections:**
- API Endpoint table: Added `/api/upload` endpoint
- Input File Handling section with 3 input types
- Uploading User Files subsection with curl/http_request examples
- Handling Compressed Archives subsection
- Example 0: Upload and Load User's h5ad File
- Error Handling: Added Upload Failed section
- Supported Platforms table: Added `h5ad` platform

**New workflow:**
```
User uploads file → /api/upload → get path → use path as value parameter
```

### 2. Tool Implementation (spatial_transcriptomics_tool.py)

**Added `_load_h5ad()` method:**
```python
def _load_h5ad(self, file_path: str) -> Any:
    """Load pre-processed AnnData h5ad file (user uploaded)."""
    import anndata as ad
    adata = ad.read_h5ad(file_path)
    logger.info(f"Loaded h5ad file: {adata.n_obs} cells, {adata.n_vars} genes")
    if "spatial" not in adata.obsm:
        logger.warning("No spatial coordinates found in h5ad file")
    return adata
```

**Auto-detection logic:**
```python
if data_dir.endswith('.h5ad') and platform != 'h5ad':
    logger.info(f"Auto-detecting h5ad file, setting platform to 'h5ad'")
    platform = 'h5ad'
```

**Updated supported_platforms:**
```python
supported_platforms = ["visium", "xenium", "merscope", "slideseq", "cosmx", "stereoseq", "h5ad"]
```

### 3. Handler Update (run_server.py)

Updated docstring to include h5ad platform and clarify that data_dir can be file path.

## New Supported Platform

| Platform | Description | Input |
|----------|-------------|-------|
| `h5ad` | Pre-processed AnnData files | Single .h5ad file (user uploaded) |

## Usage Flow

### Before Modification
```
User: "Load my Visium data"
System: Requires Space Ranger output directory structure
Result: Users with processed h5ad files cannot use this skill
```

### After Modification
```
User: "I uploaded my spatial data file (h5ad)"
1. Upload file → /api/upload → {"path": "./tmp/uploads/xxx.h5ad"}
2. Call API with value="./tmp/uploads/xxx.h5ad", query="h5ad"
3. Tool auto-detects .h5ad extension OR user specifies platform
4. _load_h5ad() reads AnnData directly
5. Returns loaded data with spatial info
```

## API Parameters

| Parameter | Field | New Behavior |
|-----------|-------|--------------|
| data_dir | value | Can now be h5ad file path OR directory |
| platform | query | Can be "h5ad" (auto-detected if .h5ad extension) |

## Testing

When deployed, test with:
```bash
# Upload file
curl -X POST "http://.../api/upload" -F "file=@test.h5ad"

# Load uploaded h5ad
curl -X POST "http://.../run_pipeline/" \
  -d '{"task": "spatial_transcriptomics_loading", "value": "./tmp/uploads/xxx.h5ad"}'
```

## Notes

1. Auto-detection: If file path ends with `.h5ad`, platform automatically set to "h5ad"
2. Spatial coordinates: Tool warns if `obsm['spatial']` missing
3. No modification to output format - still produces h5ad or zarr

## Deployment Required

Changes require redeployment of:
- open_biomed/tools/spatial_transcriptomics_tool.py
- open_biomed/scripts/run_server.py