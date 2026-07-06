---
name: single-cell-multi-omics-data-harmonization
description: >
  Multi-omics data harmonization for joint integration analysis.
  Use this skill when:
  (1) Harmonizing RNA-seq, proteomics, methylation datasets,
  (2) Applying per-assay normalization and cross-assay batch correction,
  (3) Aligning feature IDs across different omics assays,
  (4) Handling missing values before downstream integration.
license: MIT
category: bioinformatics
tags: [multi-omics, data-harmonization, batch-correction, normalization, integration]
---

# Multi-Omics Data Harmonization

Prepare your RNA-seq, proteomics, methylation, and other omics datasets for
joint integration by applying per-assay normalization, cross-assay batch
correction, feature ID alignment, and missing value handling.

This is **Step 1** of the multi-omics integration pipeline — all downstream
skills (MOFA, DIABLO, SNF) depend on clean, consistently scaled output from
this step.

---

## What it does

1. Loads each omics layer into a unified container (MultiAssayExperiment in R; MuData in Python)
2. Applies the correct normalization strategy per data type:
   - RNA-seq counts → VST (DESeq2)
   - Proteomics LFQ intensity → log2 + median centering
   - Methylation β-values → M-value transformation: log2(β / 1−β)
   - ATAC-seq peaks → log1p(CPM); miRNA → log2(CPM + 1)
3. Generates PCA plots **before** batch correction to visualize batch structure
4. Applies ComBat batch correction across assays, preserving biological condition signal
5. Generates PCA plots **after** correction to confirm batch removal
6. Maps protein UniProt IDs and methylation probe IDs to HGNC gene symbols via Ensembl BioMart
7. Filters features with > 30% missing values, then imputes remaining NAs with MinProb
8. Z-scores all features and exports as `.rds`, `.h5mu`, and `.csv` for all downstream tools

---

## Why this exists

If you ask a general AI to "prepare my multi-omics data for integration," it will:

- Apply the same normalization to all data types (wrong — RNA counts need VST; methylation needs M-value transformation, not log2)
- Run ComBat without checking for batch–condition confounding, silently removing biological signal
- Skip feature ID alignment, leaving protein UniProt IDs unmapped to gene symbols
- Not generate PCA plots to verify batch correction worked
- Export data in a format incompatible with MOFA, DIABLO, or SNF

This skill encodes the correct methodological decisions:

- Uses VST for RNA, log2+median centering for protein, and M-value for methylation — each chosen for statistical properties of that data type
- Checks `table(Batch, Condition)` before ComBat to detect confounding
- Aligns all features to HGNC gene symbols via Ensembl BioMart for cross-omics compatibility
- Filters high-missingness features before imputation to avoid noise amplification
- Exports in both R (`.rds`) and Python (`.h5mu`) formats for full downstream flexibility

---

## Reference Methods

**Normalization:**

| Data type | Method | Reason |
|---|---|---|
| RNA-seq raw counts | VST (DESeq2) | Stabilizes variance across expression range |
| Proteomics LFQ | log2 + median centering | Removes systematic MS run shifts |
| Methylation β-value | M-value: log2(β / 1−β) | Homoscedastic; better for linear models |
| ATAC-seq peaks | log1p(CPM) | Accounts for library size |
| miRNA counts | log2(CPM + 1) | Handles sparse counts |

**Batch correction:** ComBat (Johnson et al., 2007) — parametric empirical Bayes, robust for small sample sizes. Requires ≥ 2 samples per batch per condition.

**Feature alignment:** Ensembl BioMart REST API — maps UniProt accessions and Illumina probe IDs to HGNC gene symbols for cross-omics feature matching.

**Missing value imputation:** MinProb — samples replacement values from a low-intensity distribution, appropriate for proteomics data Missing Not At Random (MNAR) patterns.

---

## Handler API

Call the OpenBioMed API for multi-omics harmonization tasks.

**Base URL**: `${OPENBIOMED_API_BASE_URL}` (resolved in order: env var → Docker default → local `http://127.0.0.1:8095`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_pipeline/` | POST | Run harmonization operations |

### Operations

| Operation | Description | Required Parameters |
|-----------|-------------|---------------------|
| `load` | Load CSV files into MuData container | `data_files`, `sample_meta` |
| `normalize` | Apply per-data-type normalization | `mdata` (from load result) |
| `batch_correct` | ComBat batch correction with PCA comparison | `mdata`, `batch_column`, `condition_column` |
| `align_ids` | Map feature IDs to HGNC gene symbols | `mdata` |
| `impute` | MinProb missing value imputation | `mdata`, `missing_threshold` |
| `scale_export` | Z-score and export harmonized data | `mdata`, `export_format` |
| `full_pipeline` | Complete workflow | `data_files`, `sample_meta`, `data_types` |

### Key Methodological Decisions (from original skill)

1. **Normalization per data type**: Each assay requires a specific normalization method — not the same method for all
2. **PCA before/after correction**: Visualize batch structure before and confirm removal after
3. **Check batch-condition confounding**: If all disease samples are in one batch, skip ComBat to preserve biological signal
4. **Feature ID alignment**: Map UniProt IDs to HGNC symbols for cross-omics compatibility
5. **Missing value threshold**: Filter features with >30% missing before imputation to avoid noise amplification

### API Examples

#### Load Multi-Omics Data

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "multiomics_harmonization",
    "operation": "load",
    "data_files": {
      "rna": "path/to/rnaseq_counts.csv",
      "protein": "path/to/proteomics.csv",
      "methylation": "path/to/methylation_beta.csv"
    },
    "sample_meta": "path/to/sample_metadata.csv",
    "data_types": {
      "rna": "counts",
      "protein": "lfq",
      "methylation": "beta"
    }
  }'
```

**Response**:
```json
{
  "status": "success",
  "mdata_file": "./tmp/raw_mudata_xxx.h5mu",
  "n_assays": 3,
  "n_common_samples": 92,
  "batch_info": {"Batch1": 46, "Batch2": 46},
  "message": "Loaded 3 assays with 92 common samples"
}
```

#### Batch Correction with PCA Comparison

The `batch_correct` operation automatically:
- Checks batch-condition confounding before applying ComBat
- Generates PCA before correction to visualize batch structure
- Applies ComBat preserving biological signal (using condition as covariate)
- Generates PCA after correction to confirm batch removal

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "multiomics_harmonization",
    "operation": "batch_correct",
    "mdata": "<mdata_from_previous_step>",
    "batch_column": "Batch",
    "condition_column": "Condition"
  }'
```

**Response** (includes PCA comparison data):
```json
{
  "status": "success",
  "batch_correction_summary": {
    "rna": "ComBat (batch=Batch, covariates=[Condition])",
    "protein": "ComBat (batch=Batch, covariates=[Condition])"
  },
  "pca_comparison": {
    "rna_before": {"PC1": [...], "PC2": [...]},
    "rna_after": {"PC1": [...], "PC2": [...]}
  },
  "confounding_check": "passed - batch and condition are not confounded",
  "message": "Batch correction applied. PCA shows batch variance reduced."
}
```

#### Run Full Harmonization Pipeline

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "multiomics_harmonization",
    "operation": "full_pipeline",
    "data_files": {
      "rna": "path/to/rnaseq_counts.csv",
      "protein": "path/to/proteomics.csv"
    },
    "sample_meta": "path/to/sample_metadata.csv",
    "data_types": {
      "rna": "counts",
      "protein": "lfq"
    },
    "batch_column": "Batch",
    "condition_column": "Condition",
    "missing_threshold": 0.30,
    "export_format": "both"
  }'
```

**Response**:
```json
{
  "status": "success",
  "load": {"n_assays": 2, "n_samples": 92},
  "normalization": {
    "rna": "normalize_total(1e6) + log1p",
    "protein": "log2 + median centering"
  },
  "batch_correction": {
    "rna": "ComBat (batch=Batch, covariates=[Condition]) — PC1 batch variance: 34.2% → 2.1%",
    "protein": "ComBat (batch=Batch, covariates=[Condition]) — PC1 batch variance: 28.7% → 3.4%"
  },
  "id_mapping": {
    "protein": "UniProt→HGNC: 3,847 → 3,612 mapped (235 unmapped)"
  },
  "imputation": {
    "protein": "MinProb: 23 features filtered (>30% missing), 2.1% imputed"
  },
  "export_files": {
    "h5mu": "./tmp/harmonized_multiomics_xxx.h5mu",
    "rna_csv": "./tmp/harmonized_rna_xxx.csv",
    "protein_csv": "./tmp/harmonized_protein_xxx.csv"
  },
  "message": "Full pipeline completed. 3 files exported"
}
```

### Data Type Specifications

| Data Type | Normalization Method | Description |
|-----------|---------------------|-------------|
| `counts` | normalize_total + log1p | RNA-seq raw counts |
| `lfq` | log2 + median centering | Proteomics LFQ intensity |
| `beta` | M-value transformation | Methylation β-values |
| `peak_counts` | log1p(CPM) | ATAC-seq peaks |
| `mirna_counts` | log2(CPM + 1) | miRNA counts |

### Interpretation Guide

- **PCA before correction**: Samples should separate by Batch on PC1/PC2 if batch effect present — confirms ComBat is necessary
- **PCA after correction**: Batch structure should disappear; samples should separate by Condition instead
- **If batch and condition are confounded**: The tool will detect this and skip ComBat for affected assays
- **Protein missingness > 30%**: Features exceeding this threshold are filtered, not imputed

---

## Usage (R)

```r
library(MultiAssayExperiment)
library(DESeq2)
library(sva)
library(biomaRt)
library(ggplot2)

# ── Step 1: Build unified container ──────────────────────────────────────────
rna         <- as.matrix(read.csv("rnaseq_counts.csv",    row.names = 1))
protein     <- as.matrix(read.csv("proteomics.csv",        row.names = 1))
methylation <- as.matrix(read.csv("methylation_beta.csv", row.names = 1))
sample_info <- read.csv("sample_metadata.csv",             row.names = 1)

mae <- MultiAssayExperiment(
  experiments = ExperimentList(
    RNA         = SummarizedExperiment(assays  = list(counts    = rna),
                                        colData = sample_info),
    Protein     = SummarizedExperiment(assays  = list(intensity = protein),
                                        colData = sample_info),
    Methylation = SummarizedExperiment(assays  = list(beta      = methylation),
                                        colData = sample_info)
  ),
  colData = sample_info
)
mae <- intersectColumns(mae)   # keep only samples present in ALL assays
cat("Samples in all assays:", ncol(mae[[1]]), "\n")

# ── Step 2: Normalize per assay ───────────────────────────────────────────────
dds     <- DESeqDataSet(mae[["RNA"]], design = ~1)
vst_rna <- assay(vst(dds, blind = TRUE))

log2_prot <- log2(assay(mae[["Protein"]]))
log2_prot[is.infinite(log2_prot)] <- NA
norm_prot <- sweep(log2_prot, 2,
                   apply(log2_prot, 2, median, na.rm = TRUE) -
                   median(log2_prot, na.rm = TRUE))

beta   <- pmin(pmax(assay(mae[["Methylation"]]), 0.001), 0.999)
m_vals <- log2(beta / (1 - beta))

# ── Step 3: PCA before correction ────────────────────────────────────────────
pca_plot <- function(mat, meta, title) {
  df <- as.data.frame(prcomp(t(mat))$x[, 1:2])
  df$Batch <- meta$Batch; df$Condition <- meta$Condition
  ggplot(df, aes(PC1, PC2, color = Batch, shape = Condition)) +
    geom_point(size = 3) + ggtitle(title) + theme_bw()
}
p_before <- pca_plot(vst_rna, colData(mae), "RNA — before correction")

# ── Step 4: Batch correction ──────────────────────────────────────────────────
# Always check confounding first
print(table(colData(mae)$Batch, colData(mae)$Condition))

batch          <- colData(mae)$Batch
mod            <- model.matrix(~ Condition, data = as.data.frame(colData(mae)))
corrected_rna  <- ComBat(vst_rna,   batch = batch, mod = mod)
corrected_prot <- ComBat(norm_prot, batch = batch, mod = mod)
corrected_meth <- ComBat(m_vals,    batch = batch, mod = mod)

p_after <- pca_plot(corrected_rna, colData(mae), "RNA — after correction")
ggsave("figures/pca_before_after.pdf",
       gridExtra::grid.arrange(p_before, p_after, ncol = 2),
       width = 12, height = 5)

# ── Step 5: Feature ID alignment ──────────────────────────────────────────────
ensembl <- useEnsembl("genes", dataset = "hsapiens_gene_ensembl")
id_map  <- getBM(attributes = c("uniprotswissprot", "hgnc_symbol"),
                  filters    = "uniprotswissprot",
                  values     = rownames(corrected_prot), mart = ensembl)
rownames(corrected_prot) <- id_map$hgnc_symbol[
  match(rownames(corrected_prot), id_map$uniprotswissprot)]
corrected_prot <- corrected_prot[!is.na(rownames(corrected_prot)), ]

# ── Step 6: Missing value handling ───────────────────────────────────────────
corrected_prot <- corrected_prot[rowMeans(is.na(corrected_prot)) < 0.30, ]

impute_minprob <- function(mat) {
  for (j in seq_len(ncol(mat))) {
    nas <- is.na(mat[, j])
    if (any(nas)) {
      q01 <- quantile(mat[, j], 0.01, na.rm = TRUE)
      mat[nas, j] <- rnorm(sum(nas), mean = q01, sd = abs(q01) * 0.1)
    }
  }; mat
}
corrected_prot <- impute_minprob(corrected_prot)

# ── Step 7: Scale and export ──────────────────────────────────────────────────
scale_mat  <- function(mat) t(scale(t(mat)))
harmonized <- list(
  RNA         = scale_mat(corrected_rna),
  Protein     = scale_mat(corrected_prot),
  Methylation = scale_mat(corrected_meth),
  sample_meta = as.data.frame(colData(mae))
)
saveRDS(harmonized, "harmonized_multiomics.rds")
write.csv(harmonized$RNA,         "harmonized_rna.csv")
write.csv(harmonized$Protein,     "harmonized_protein.csv")
write.csv(harmonized$Methylation, "harmonized_methylation.csv")
```

## Usage (Python — muon)

```python
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from combat.pycombat import pycombat

adata_rna  = sc.read_csv("rnaseq_counts.csv").T
adata_prot = sc.read_csv("proteomics.csv").T
meta       = pd.read_csv("sample_metadata.csv", index_col=0)

for a in [adata_rna, adata_prot]:
    a.obs = meta.loc[a.obs_names]

sc.pp.normalize_total(adata_rna, target_sum=1e6)
sc.pp.log1p(adata_rna)

X = np.log2(adata_prot.X + 1)
X -= np.median(X, axis=0) - np.median(X)
adata_prot.X = X

batch = meta.loc[adata_rna.obs_names, "Batch"]
for adata in [adata_rna, adata_prot]:
    df = pd.DataFrame(adata.X.T, columns=adata.obs_names,
                      index=adata.var_names)
    adata.X = pycombat(df, batch).T.values

mdata = mu.MuData({"rna": adata_rna, "prot": adata_prot})
mu.pp.intersect_obs(mdata)
mdata.write("harmonized_multiomics.h5mu")
```

---

## Example Output

```
Multi-Omics Data Harmonization
===============================
Input assays:   RNA | Protein | Methylation
Common samples: 92 (of 97 total — 5 missing at least one assay)

Normalization applied:
  RNA:         VST (DESeq2)         — 18,432 features
  Protein:     log2 + median center — 3,847 features
  Methylation: M-value              — 11,209 features

Batch correction (ComBat, 2 batches):
  RNA  — PC1 batch variance: 34.2% → 2.1%   ✓ corrected
  Protein — PC1 batch variance: 28.7% → 3.4% ✓ corrected

Feature ID alignment:
  Protein: 3,847 UniProt IDs → 3,612 HGNC symbols
  (235 removed: unmapped or duplicate)

Missing value summary (protein):
  Filtered (>30% missing): 3,612 → 3,589 features retained
  Imputed (MinProb):        0% missing after imputation

Exported:
  harmonized_multiomics.rds
  harmonized_rna.csv         (18,432 × 92)
  harmonized_protein.csv     ( 3,589 × 92)
  harmonized_methylation.csv (11,209 × 92)
  figures/pca_before_after.pdf
```

---

## Interpretation Guide

- **PCA before correction**: Samples should separate by Batch on PC1/PC2 if a batch effect is present — this confirms ComBat is necessary
- **PCA after correction**: Batch structure should disappear; samples should now separate by Condition instead
- **If batch and condition are confounded** (e.g., all disease samples in Batch 1): skip ComBat — it will remove the biological signal you want to keep
- **Protein missingness > 30%**: features exceeding this threshold are filtered, not imputed — imputing very sparse features introduces more noise than signal
- **M-value vs β-value**: M-values are statistically preferable for downstream modelling (homoscedastic); β-values (0–1 range) are more interpretable for visualization — keep both if needed

---

## Citation

If you use this skill in a publication, please cite:

- Johnson, W.E. et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.
- Ramos, M. et al. (2017). Software for the integration of multiomics experiments in Bioconductor. *Cancer Research*, 77(21), e39–e42. (MultiAssayExperiment)
- Love, M.I. et al. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15, 550.
