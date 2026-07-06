---
name: single-cell-proteomics-peptide-identification
description: >
  Peptide and protein identification from MS2 spectra using MSFragger/Philosopher.
  Use this skill when:
  (1) Searching MS2 spectra against protein sequence database,
  (2) Applying target-decoy FDR filtering at PSM and protein levels,
  (3) Performing protein inference with parsimony grouping,
  (4) Preparing protein database with decoys and contaminants.
license: MIT
category: bioinformatics
tags: [proteomics, mass-spectrometry, peptide-identification, msfragger, philosopher, fdr]
---

# Peptide and Protein Identification

Search MS2 spectra against a protein sequence database to identify peptides
and proteins in your sample. Apply target-decoy FDR filtering to control
false discovery rate at both PSM and protein levels.

This is **Step 2** of the proteomics pipeline — takes centroided mzML from
Step 1, produces PSM tables and protein groups for Step 3 (quantification).

---

## What it does

1. Prepares the protein database: appends decoy sequences (reverse or scrambled) and common contaminants
2. Configures search parameters: enzyme specificity, variable and fixed modifications, mass tolerances
3. Runs database search with MSFragger (recommended) or Comet
4. Applies PSM-level FDR filtering using Percolator rescoring or classical target-decoy approach
5. Performs protein inference with parsimony principle to resolve shared peptides
6. Filters protein groups to 1% FDR
7. Exports results as TSV tables, pepXML, and mzIdentML
8. Generates summary statistics: number of PSMs, unique peptides, and protein groups

---

## Why this exists

If you ask a general AI to "identify peptides in my MS data," it will:

- Not explain the target-decoy strategy or why it is required for FDR estimation
- Use incorrect MSFragger command-line flags (the CLI changed between v3 and v4)
- Skip protein inference entirely, leaving only peptide-level results
- Not distinguish between PSM FDR, peptide FDR, and protein FDR — applying only one threshold
- Not add contaminant sequences to the database, leading to misidentification of common lab proteins

This skill encodes the correct methodological decisions:

- Always appends a decoy database before searching (reversed sequences at minimum)
- Adds cRAP contaminant database (116 common laboratory contaminants)
- Distinguishes PSM FDR (1%) from protein FDR (1%) and applies both
- Uses parsimony protein grouping to handle shared peptides correctly
- Explains parameter choices for common modifications (oxidation M, carbamidomethyl C)

---

## Reference Methods

**Target-decoy strategy:** Every real protein sequence (target) is paired with a scrambled/reversed sequence (decoy). Because decoys cannot exist in the sample, PSM matches to decoys are false positives. The FDR at score threshold T is estimated as: `FDR = (# decoy PSMs above T) / (# target PSMs above T)`.

**MSFragger** (Kong et al., 2017) uses a fragment ion indexing approach for ultra-fast closed and open (mass-tolerant) database search. Recommended for standard DDA proteomics; also supports DIA via MSBooster + DIA-NN integration.

**Comet** (Eng et al., 2013) is a well-established open-source search engine. Slower than MSFragger for large datasets but widely supported and parameter-compatible with many downstream tools.

**Percolator** (Käll et al., 2007) uses semi-supervised machine learning to rescore PSMs, improving sensitivity at a given FDR threshold compared to classical score cutoffs.

**Protein inference (parsimony):** When a peptide is shared between multiple proteins, the parsimony principle selects the minimum set of proteins that explains all observed peptides. Implemented in Philosopher's `proteinprophet` module.

---

## Handler API

Call the OpenBioMed API for peptide and protein identification tasks.

**Base URL**: `${OPENBIOMED_API_BASE_URL}` (resolved in order: env var → Docker default → local `http://127.0.0.1:8095`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_pipeline/` | POST | Run identification operations |

### Operations

| Operation | Description | Required Parameters |
|-----------|-------------|---------------------|
| `prepare_database` | Prepare protein database with decoys and contaminants | `organism` |
| `search` | Run MSFragger database search | `mzml_files`, `database_file` |
| `validate` | Run Philosopher validation (PeptideProphet, ProteinProphet, filter) | `mzml_files`, `database_file` |
| `full_pipeline` | Complete workflow (prepare + search + validate) | `mzml_files`, `organism` |
| `parse_results` | Parse TSV output files into structured data | `output_dir` |

### Key Methodological Decisions (from original skill)

1. **Target-decoy strategy**: Every target sequence must have a decoy pair — decoy matches are false positives used for FDR estimation: `FDR = (# decoy hits) / (# target hits)`
2. **cRAP contaminants**: Always add 116 common laboratory contaminants to avoid misidentifying keratin, trypsin, etc.
3. **Three-level FDR**: Apply **PSM FDR**, **peptide FDR**, and **protein FDR** separately — they are NOT equivalent
4. **Parsimony protein grouping**: When peptides are shared between proteins, select the minimum protein set that explains all observations
5. **Modification choices**: Carbamidomethyl C (fixed, from alkylation), Oxidation M (variable, common artifact)

### API Examples

#### Prepare Database (UniProt + cRAP + Decoys)

The `prepare_database` operation automatically:
- Downloads UniProt reference proteome for the specified organism
- Downloads cRAP contaminant database (116 common lab proteins)
- Appends reversed decoy sequences with `DECOY_` prefix

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "peptide_identification",
    "operation": "prepare_database",
    "organism": "human",
    "output_dir": "./tmp/"
  }'
```

**Response**:
```json
{
  "status": "success",
  "database_file": "./tmp/workspace_xxx/human_combined_td.fasta",
  "uniprot_file": "./tmp/human_uniprot_xxx.fasta",
  "crap_file": "./tmp/crap_xxx.fasta",
  "n_target_proteins": 20434,
  "n_decoy_proteins": 20434,
  "n_contaminants": 116,
  "message": "Database prepared: 20434 target + 20434 decoy + 116 contaminants"
}
```

#### Run MSFragger Search

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "peptide_identification",
    "operation": "search",
    "mzml_files": ["path/to/sample1.mzML", "path/to/sample2.mzML"],
    "database_file": "path/to/database_td.fasta",
    "output_dir": "./tmp/",
    "search_params": {
      "precursor_mass_tolerance": 20,
      "fragment_mass_tolerance": 20,
      "enzyme": "Trypsin",
      "missed_cleavages": 2
    }
  }'
```

**Response**:
```json
{
  "status": "success",
  "params_file": "./tmp/fragger_xxx.params",
  "pepxml_files": ["./tmp/sample1.pepXML", "./tmp/sample2.pepXML"],
  "n_pepxml": 2,
  "message": "MSFragger search completed. Generated 2 pepXML files"
}
```

#### Run Validation (PeptideProphet + ProteinProphet + FDR Filter)

The `validate` operation automatically:
- Runs PeptideProphet for PSM validation
- Runs iProphet to combine across runs
- Runs ProteinProphet for protein inference (parsimony grouping)
- Applies FDR filtering at PSM, peptide, and protein levels

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "peptide_identification",
    "operation": "validate",
    "mzml_files": ["path/to/sample1.mzML", "path/to/sample2.mzML"],
    "database_file": "path/to/database_td.fasta",
    "output_dir": "./tmp/",
    "fdr_threshold": 0.01
  }'
```

**Response**:
```json
{
  "status": "success",
  "pepxml_files": ["./tmp/sample1.pepXML", "./tmp/sample2.pepXML"],
  "tsv_files": {
    "psm": "./tmp/psm.tsv",
    "peptide": "./tmp/peptide.tsv",
    "protein": "./tmp/protein.tsv",
    "ion": "./tmp/ion.tsv"
  },
  "fdr_threshold": 0.01,
  "message": "Validation completed. FDR filtered at 1% at all levels"
}
```

#### Run Full Pipeline

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "peptide_identification",
    "operation": "full_pipeline",
    "mzml_files": ["path/to/sample1.mzML", "path/to/sample2.mzML"],
    "organism": "human",
    "output_dir": "./tmp/",
    "fdr_threshold": 0.01
  }'
```

**Response**:
```json
{
  "status": "success",
  "database_file": "./tmp/workspace_xxx/human_combined_td.fasta",
  "n_target_proteins": 20434,
  "n_decoy_proteins": 20434,
  "n_contaminants": 116,
  "tsv_files": {
    "psm": "./tmp/psm.tsv",
    "peptide": "./tmp/peptide.tsv",
    "protein": "./tmp/protein.tsv"
  },
  "summary": {
    "n_psms": 87341,
    "n_peptides": 24628,
    "n_proteins": 4204,
    "charge_distribution": {"1": 3.2, "2": 41.8, "3": 38.1, "4+": 16.9},
    "missed_cleavage_distribution": {"0": 81.8, "1": 15.9, "2": 2.3}
  },
  "message": "Full pipeline completed. 87341 PSMs identified at 1% FDR"
}
```

### Search Parameters

Key search parameters for MSFragger:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `precursor_mass_tolerance` | 20 | Precursor mass tolerance (ppm) |
| `fragment_mass_tolerance` | 20 | Fragment mass tolerance (ppm) |
| `enzyme` | Trypsin | Enzyme specificity |
| `missed_cleavages` | 2 | Allowed missed cleavages |
| `fixed_mods` | C+57.02146 | Fixed modifications (carbamidomethylation from alkylation) |
| `variable_mods` | M+15.99491 | Variable modifications (oxidation, common artifact) |

### Prerequisites

- **MSFragger.jar**: Download from https://github.com/Nesvilab/MSFragger/releases
- **philosopher.jar**: Download from https://github.com/Nesvilab/philosopher/releases
- **Java 11+**: Required for MSFragger 4.x

### Interpretation Guide

- **PSM vs peptide ratio**: Ratio ~3.5:1 is typical. < 1.5 suggests shallow run; > 10 suggests over-sampling
- **Missed cleavage rate**: MC=0 > 75% indicates good digestion. MC=2 > 10% suggests incomplete digestion
- **Protein FDR types**: PSM FDR ≠ protein FDR. "Picked protein FDR" is more conservative for large datasets
- **Charge state z=1**: High z=1 fraction (> 15%) indicates in-source fragmentation or non-tryptic peptides

---

## Usage

### Part A — Database preparation

```bash
# Download UniProt human reference proteome (canonical + isoforms)
wget -O human_uniprot.fasta \
  "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(proteome:UP000005640)"

# Download cRAP contaminants (116 common laboratory proteins)
wget -O crap.fasta \
  "https://www.thegpm.org/crap/caRaP.fasta"

# Append decoys using philosopher (reverses sequences, adds DECOY_ prefix)
philosopher workspace --init
philosopher database --reviewed --contam --custom crap.fasta \
  --prefix "DECOY_" human_uniprot.fasta

# Result: database file ending in _td.fasta (target + decoy)
ls *.fasta
# human_uniprot_crap_td.fasta   ← use this for search
```

### Part B — MSFragger database search

```bash
# MSFragger parameter file (fragger.params) — key settings
cat > fragger.params << 'EOF'
database_name = human_uniprot_crap_td.fasta
num_threads = 16

# Enzyme
search_enzyme_name = Trypsin
search_enzyme_cut_after = KR
search_enzyme_no_cut_before = P
allowed_missed_cleavage = 2

# Mass tolerances (Orbitrap values)
precursor_mass_tolerance = 20          # ppm
precursor_mass_units = 1               # 1 = ppm
fragment_mass_tolerance = 20           # ppm
fragment_mass_units = 1                # 1 = ppm

# Fixed modifications
add_C_cysteine = 57.02146              # carbamidomethylation

# Variable modifications (up to 3 on a single peptide)
variable_mod_01 = 15.99491 M 3         # oxidation of Met
variable_mod_02 = 42.01057 n 1         # N-terminal acetylation

# Peptide length/mass filters
digest_min_length = 7
digest_max_length = 50
precursor_charge = 1 4

output_format = pepXML
EOF

# Run MSFragger (requires Java 11+)
java -Xmx64g -jar MSFragger-4.0.jar fragger.params \
  run1.mzML run2.mzML run3.mzML
# Output: one .pepXML file per mzML input
```

### Part C — PSM validation with Percolator (via Philosopher)

```bash
# Use Philosopher to run PeptideProphet + Percolator rescoring
philosopher peptideprophet \
  --database human_uniprot_crap_td.fasta \
  --decoy DECOY_ \
  --ppm \
  --accmass \
  run1.pepXML run2.pepXML run3.pepXML

# Combine results across runs
philosopher iprophet \
  --output combined.pep.xml \
  interact-*.pep.xml

# Protein inference (parsimony)
philosopher proteinprophet \
  --output proteinprophet.prot.xml \
  combined.pep.xml

# Filter at 1% PSM, peptide, and protein FDR
philosopher filter \
  --psm   0.01 \
  --pep   0.01 \
  --prot  0.01 \
  --picked \
  --tag DECOY_ \
  --razor

# Export final TSV tables
philosopher report
# Outputs: psm.tsv, peptide.tsv, protein.tsv, ion.tsv
```

### Part D — Parse and inspect results in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# ── Load PSM table ────────────────────────────────────────────────────────────
psm_df = pd.read_csv("psm.tsv", sep="\t")
pep_df = pd.read_csv("peptide.tsv", sep="\t")
prot_df = pd.read_csv("protein.tsv", sep="\t")

print("=== Identification Summary ===")
print(f"PSMs (1% FDR):          {len(psm_df):,}")
print(f"Unique peptides:        {pep_df['Peptide'].nunique():,}")
print(f"Protein groups (1% FDR): {len(prot_df):,}")

# ── Charge state distribution ─────────────────────────────────────────────────
charge_counts = psm_df["Charge"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(charge_counts.index.astype(str), charge_counts.values,
       color="#4DBBD5", edgecolor="white")
ax.set_xlabel("Precursor charge state")
ax.set_ylabel("PSM count")
ax.set_title("Charge state distribution")
plt.tight_layout()
plt.savefig("figures/charge_distribution.pdf")

# ── Missed cleavages distribution ─────────────────────────────────────────────
mc_counts = psm_df["Missed Cleavages"].value_counts().sort_index()
print("\nMissed cleavages:")
for mc, count in mc_counts.items():
    pct = count / len(psm_df) * 100
    print(f"  MC={mc}: {count:,} PSMs ({pct:.1f}%)")

# ── Protein coverage ──────────────────────────────────────────────────────────
top_proteins = prot_df[["Protein", "Gene", "Coverage",
                          "Unique Peptides", "Spectral Count"]].head(20)
print("\nTop 20 proteins by spectral count:")
print(top_proteins.to_string(index=False))

# ── Score distribution (PSMs) ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(psm_df["Hyperscore"], bins=50, color="#E64B35", alpha=0.7)
axes[0].set_xlabel("Hyperscore"); axes[0].set_title("PSM score distribution")

axes[1].hist(psm_df["PeptideProphet Probability"], bins=50,
             color="#00A087", alpha=0.7)
axes[1].set_xlabel("PeptideProphet probability")
axes[1].set_title("Posterior probability distribution")

plt.tight_layout()
plt.savefig("figures/score_distributions.pdf")
plt.close()
```

### Part E — Comet (alternative search engine)

```bash
# Comet parameter file (comet.params)
cat > comet.params << 'EOF'
database_name = human_uniprot_crap_td.fasta
decoy_search = 1                   # 1 = append reversed decoys
num_threads = 16
output_pepxmlfile = 1
output_txtfile = 1

# Enzyme
search_enzyme_number = 1           # 1 = Trypsin
num_enzyme_termini = 2
allowed_missed_cleavage = 2

# Tolerances (Orbitrap)
peptide_mass_tolerance = 20.0
peptide_mass_units = 2             # 2 = ppm
fragment_bin_tol = 0.02
fragment_bin_offset = 0.0

# Modifications
add_C_cysteine = 57.0215           # fixed: carbamidomethyl
variable_mod01 = 15.9949 M 0 3 -1  # variable: oxidation
EOF

# Run Comet
comet -P comet.params run1.mzML run2.mzML run3.mzML
```

---

## Example Output

```
Peptide and Protein Identification
====================================
Search engine: MSFragger 4.0
Database: UniProt human (20,434 proteins) + cRAP (116) + decoys
Runs searched: 3 mzML files

Identification results (1% FDR at all levels):
  PSMs:                  87,341
  Unique peptides:       24,628
  Unique proteins:        4,891
  Protein groups:         4,204

Missed cleavage distribution:
  MC=0: 71,420 PSMs (81.8%)
  MC=1: 13,891 PSMs (15.9%)
  MC=2:  2,030 PSMs  (2.3%)

Charge distribution:
  z=1:    3.2%    z=2:  41.8%
  z=3:   38.1%    z=4+: 16.9%

Score thresholds applied:
  PSM FDR:     1.0% (PeptideProphet ≥ 0.90)
  Peptide FDR: 1.0%
  Protein FDR: 1.0% (picked protein FDR)

Exported:
  psm.tsv       (87,341 rows)
  peptide.tsv   (24,628 rows)
  protein.tsv   ( 4,204 rows)
  figures/charge_distribution.pdf
  figures/score_distributions.pdf
```

---

## Interpretation Guide

- **PSM count vs. unique peptide count**: The ratio (here ~3.5:1) reflects average spectral redundancy. Ratio < 1.5 suggests a shallow run; ratio > 10 suggests over-sampling of abundant peptides
- **Missed cleavage rate**: MC=0 > 75% indicates good digestion efficiency. MC=2 > 10% may indicate incomplete digestion — check trypsin amount, digestion time, and protein denaturation
- **Protein FDR types**: PSM FDR ≠ protein FDR. Always apply protein-level FDR separately. "Picked protein FDR" (Savitski et al., 2015) is more conservative and recommended for large datasets
- **Charge state z=1**: High z=1 fraction (> 15%) can indicate in-source fragmentation, short peptides, or non-tryptic cleavage
- **PeptideProphet probability**: Values above 0.90–0.95 are typically required for 1% PSM FDR. Very low probability peaks near 0.5 indicate poor spectral quality
- **Protein groups vs. unique proteins**: Protein groups consolidate proteins that share all detected peptides. The number of groups is always ≤ number of unique protein accessions

---

## Citation

If you use this skill in a publication, please cite:

- Kong, A.T. et al. (2017). MSFragger: ultrafast and comprehensive peptide identification in mass spectrometry–based proteomics. *Nature Methods*, 14(5), 513–520.
- Eng, J.K. et al. (2013). Comet: an open-source MS/MS sequence database search tool. *Proteomics*, 13(1), 22–24.
- Käll, L. et al. (2007). Semi-supervised learning for peptide identification from shotgun proteomics datasets. *Nature Methods*, 4(11), 923–925. (Percolator)
- da Veiga Leprevost, F. et al. (2020). Philosopher: a versatile toolkit for shotgun proteomics data analysis. *Nature Methods*, 17(9), 869–870.
