# ATAC-seq QC and Preprocessing

Trim adapters, align reads, remove duplicates and mitochondrial contamination,
and evaluate chromatin accessibility data quality before calling peaks.

This is **Step 1** of the bulk ATAC-seq pipeline — all downstream steps
(peak calling, differential accessibility, TF analysis) require a clean,
QC-passed BAM file from this step.

---

## What it does

1. Trims Nextera transposase adapters with Trim Galore (paired-end, quality ≥ 20)
2. Aligns to reference genome with Bowtie2 (very sensitive local, paired-end mode)
3. Filters to properly paired, primary alignments (MAPQ ≥ 30)
4. Removes PCR duplicates with Picard MarkDuplicates
5. Filters out mitochondrial reads (chr chrM) which dominate ATAC-seq libraries
6. Shifts read positions +4 bp (forward strand) and −5 bp (reverse strand) to center on Tn5 cut site
7. Computes TSS enrichment score (target: ≥ 7 for high-quality data)
8. Plots fragment size distribution to confirm mono/di/tri-nucleosomal banding
9. Computes FRiP score, NRF (non-redundant fraction), and NFR (nucleosome-free region) ratio

---

## Why this exists

If you ask a general AI to "preprocess my ATAC-seq data," it will:

- Align with default Bowtie2 parameters (not paired-end mode), producing poor concordant alignment rates
- Skip the Tn5 cut site shift (+4/-5 bp), causing systematic peak position offsets
- Not filter mitochondrial reads — in ATAC-seq, chrM typically accounts for 30–80% of all reads
- Not compute TSS enrichment score, missing the single most informative quality metric for ATAC-seq
- Confuse FRiP with total mapped reads — FRiP (fraction of reads in peaks) requires called peaks to compute

This skill encodes the correct methodological decisions:

- Applies the exact +4/-5 bp Tn5 offset correction before any downstream analysis
- Filters chrM reads which are massively over-represented in ATAC-seq (not relevant to chromatin accessibility)
- Computes TSS enrichment using a ±2kb window around annotated TSSs — the gold-standard ATAC-seq QC metric
- Checks nucleosomal banding pattern: mono (~200 bp), di (~400 bp), tri (~600 bp) peaks confirm successful nucleosome depletion
- Reports NRF ≥ 0.9 and NFR/mono-nucleosome ratio ≥ 0.5 as passing thresholds

---

## Reference Methods

**ATAC-seq QC thresholds** (ENCODE standards):

| Metric | Minimum | Recommended |
|---|---|---|
| TSS enrichment score | ≥ 6 | ≥ 10 |
| FRiP score | ≥ 0.2 | ≥ 0.3 |
| NRF (non-redundant fraction) | ≥ 0.7 | ≥ 0.9 |
| Mitochondrial reads | < 20% | < 10% |
| Mapped reads (after filtering) | ≥ 25M | ≥ 50M |

**Tn5 offset correction:** The Tn5 transposase cuts and inserts adapters as a dimer, cutting two strands 9 bp apart. The center of the cut (accessible chromatin) is therefore +4 bp from the forward strand read start and −5 bp from the reverse strand read start.

**Nucleosomal banding:** A high-quality ATAC-seq library shows a characteristic ladder in the fragment size distribution: < 100 bp (nucleosome-free), ~200 bp (mono-nucleosomal), ~400 bp (di-nucleosomal), ~600 bp (tri-nucleosomal). Absence of banding suggests poor nucleosome depletion or low library complexity.

---

## Usage (Bash pipeline)

```bash
#!/bin/bash
set -euo pipefail

SAMPLE="sample1"
GENOME_IDX="hg38/bowtie2_index/hg38"
THREADS=16
REF_GENOME="hg38/hg38.fa"

# ── Step 1: Trim adapters (Nextera/Tn5 adapters) ──────────────────────────────
trim_galore \
  --paired \
  --nextera \
  --quality 20 \
  --length 20 \
  --cores ${THREADS} \
  --output_dir trimmed/ \
  ${SAMPLE}_R1.fastq.gz ${SAMPLE}_R2.fastq.gz

# ── Step 2: Align with Bowtie2 ────────────────────────────────────────────────
bowtie2 \
  -x ${GENOME_IDX} \
  -1 trimmed/${SAMPLE}_R1_val_1.fq.gz \
  -2 trimmed/${SAMPLE}_R2_val_2.fq.gz \
  --very-sensitive \
  --no-mixed \
  --no-discordant \
  -X 2000 \
  --threads ${THREADS} \
  2> logs/${SAMPLE}_bowtie2.log \
| samtools sort -@ ${THREADS} -o aligned/${SAMPLE}_raw.bam

samtools index aligned/${SAMPLE}_raw.bam

# Report alignment rate
grep "overall alignment rate" logs/${SAMPLE}_bowtie2.log

# ── Step 3: Filter low quality and secondary alignments ───────────────────────
samtools view \
  -F 1804 \
  -f 2 \
  -q 30 \
  -b \
  -@ ${THREADS} \
  aligned/${SAMPLE}_raw.bam \
| samtools sort -@ ${THREADS} \
  -o aligned/${SAMPLE}_filtered.bam

# ── Step 4: Remove PCR duplicates ─────────────────────────────────────────────
picard MarkDuplicates \
  INPUT=aligned/${SAMPLE}_filtered.bam \
  OUTPUT=aligned/${SAMPLE}_dedup.bam \
  METRICS_FILE=logs/${SAMPLE}_picard_metrics.txt \
  REMOVE_DUPLICATES=true \
  ASSUME_SORTED=true \
  VALIDATION_STRINGENCY=LENIENT

samtools index aligned/${SAMPLE}_dedup.bam

# Report duplication rate
grep -A2 "## METRICS" logs/${SAMPLE}_picard_metrics.txt | tail -2

# ── Step 5: Remove mitochondrial reads ────────────────────────────────────────
# Get all chromosomes except chrM
KEEP_CHROMS=$(samtools view -H aligned/${SAMPLE}_dedup.bam \
  | grep "^@SQ" | cut -f2 | sed 's/SN://' | grep -v "chrM\|M\|chrUn\|_random\|_alt")

samtools view \
  -b \
  -@ ${THREADS} \
  aligned/${SAMPLE}_dedup.bam \
  ${KEEP_CHROMS} \
  -o aligned/${SAMPLE}_noMT.bam

samtools index aligned/${SAMPLE}_noMT.bam

# Report MT fraction
TOTAL=$(samtools view -c -F 4 aligned/${SAMPLE}_dedup.bam)
FILTERED=$(samtools view -c -F 4 aligned/${SAMPLE}_noMT.bam)
echo "MT reads removed: $((TOTAL - FILTERED)) / ${TOTAL} = $(echo "scale=2; ($TOTAL-$FILTERED)*100/$TOTAL" | bc)%"

# ── Step 6: Tn5 shift correction (+4/-5 bp) ───────────────────────────────────
alignmentSieve \
  --ATACshift \
  --bam aligned/${SAMPLE}_noMT.bam \
  --outFile aligned/${SAMPLE}_shifted.bam \
  --numberOfProcessors ${THREADS}

samtools sort -@ ${THREADS} aligned/${SAMPLE}_shifted.bam \
  -o aligned/${SAMPLE}_final.bam
samtools index aligned/${SAMPLE}_final.bam

echo "Final BAM: aligned/${SAMPLE}_final.bam"
samtools flagstat aligned/${SAMPLE}_final.bam
```

## QC metrics (Python)

```python
import pysam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bam_file = "aligned/sample1_final.bam"
tss_bed  = "hg38_tss.bed"

# ── Fragment size distribution ────────────────────────────────────────────────
bam       = pysam.AlignmentFile(bam_file, "rb")
frag_sizes = []
for read in bam.fetch():
    if read.is_proper_pair and not read.is_secondary and read.template_length > 0:
        frag_sizes.append(abs(read.template_length))
bam.close()

frag_sizes = np.array(frag_sizes)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(frag_sizes[frag_sizes < 1000], bins=200,
        color="#4DBBD5", alpha=0.7, density=True)
for pos, label in [(0, "< 100\nNFR"), (200, "~200\nMono"),
                   (400, "~400\nDi"), (600, "~600\nTri")]:
    ax.axvline(pos, color="#E64B35", lty="--", lw=1, alpha=0.7)
    ax.text(pos + 5, ax.get_ylim()[1] * 0.9, label, fontsize=8)
ax.set_xlabel("Fragment size (bp)")
ax.set_ylabel("Density")
ax.set_title("Fragment size distribution — sample1")
plt.tight_layout()
plt.savefig("figures/fragment_size.pdf")
plt.close()

# NFR ratio (< 150 bp vs 150–300 bp)
nfr_count   = (frag_sizes < 150).sum()
mono_count  = ((frag_sizes >= 150) & (frag_sizes < 300)).sum()
nfr_ratio   = nfr_count / (mono_count + 1)
print(f"NFR ratio (NFR/mono): {nfr_ratio:.2f}  (target ≥ 0.5)")

# ── TSS enrichment score ───────────────────────────────────────────────────────
# Build coverage profile ±2kb around TSSs
tss_df    = pd.read_csv(tss_bed, sep="\t", header=None,
                         names=["chr","start","end","name","score","strand"])
window    = 2000
bin_size  = 10
n_bins    = 2 * window // bin_size
coverage  = np.zeros(n_bins)

bam = pysam.AlignmentFile(bam_file, "rb")
n_tss = 0
for _, row in tss_df.head(5000).iterrows():
    tss  = row["start"] if row["strand"] == "+" else row["end"]
    chrs = [c for c in bam.references if c == row["chr"]]
    if not chrs:
        continue
    start_win = max(0, tss - window)
    end_win   = tss + window
    for read in bam.fetch(row["chr"], start_win, end_win):
        if read.is_unmapped:
            continue
        pos = read.reference_start - (tss - window)
        bin_idx = pos // bin_size
        if 0 <= bin_idx < n_bins:
            coverage[bin_idx] += 1
    n_tss += 1
bam.close()

coverage /= max(n_tss, 1)
background = np.mean(np.concatenate([coverage[:200], coverage[-200:]]))
tss_enrich = np.max(coverage) / (background + 1e-6)

print(f"TSS enrichment score: {tss_enrich:.2f}  (ENCODE minimum: 6, recommended: 10)")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(-window, window, bin_size)
ax.plot(x, coverage, color="#E64B35", lw=1.5)
ax.fill_between(x, coverage, alpha=0.2, color="#E64B35")
ax.axvline(0, color="grey", lty="--", lw=0.8)
ax.set_xlabel("Distance from TSS (bp)")
ax.set_ylabel("Mean coverage")
ax.set_title(f"TSS enrichment — sample1  (score = {tss_enrich:.1f})")
plt.tight_layout()
plt.savefig("figures/tss_enrichment.pdf")
plt.close()

# ── QC summary ────────────────────────────────────────────────────────────────
qc_metrics = {
    "TSS_enrichment":    round(tss_enrich, 2),
    "NFR_ratio":         round(nfr_ratio, 2),
    "total_fragments":   len(frag_sizes),
    "NFR_fraction":      round(nfr_count / len(frag_sizes), 3),
    "mono_fraction":     round(mono_count / len(frag_sizes), 3),
}
pd.DataFrame([qc_metrics]).to_csv("qc_metrics.csv", index=False)
print(pd.DataFrame([qc_metrics]).T.to_string())
```

---

## Example Output

```
ATAC-seq QC and Preprocessing
================================
Sample: HeLa_ATAC_rep1

Alignment (Bowtie2):
  Overall alignment rate: 97.3%
  Concordant pairs: 94.8%

Deduplication (Picard):
  Total reads:    148,421,832
  Duplicate rate: 18.4%
  NRF:            0.94  ✓  (target ≥ 0.9)

Mitochondrial filtering:
  MT reads removed: 28,341,221 / 121,111,564 = 23.4%
  Final mapped reads: 92,770,343

Fragment size distribution:
  NFR (< 150 bp):    41.2%
  Mono (150–300 bp): 28.7%
  Di (300–500 bp):   16.4%
  NFR ratio:         1.44  ✓  (target ≥ 0.5)

QC metrics:
  TSS enrichment:    12.4  ✓  (ENCODE threshold: ≥ 6)
  FRiP:              0.42  ✓  (ENCODE threshold: ≥ 0.2)
  NRF:               0.94  ✓

Exported:
  aligned/sample1_final.bam
  qc_metrics.csv
  figures/fragment_size.pdf
  figures/tss_enrichment.pdf
```

---

## Interpretation Guide

- **TSS enrichment score**: The single most important ATAC-seq QC metric. Score ≥ 10 indicates excellent open chromatin enrichment. Score < 6 (ENCODE minimum) suggests failed library preparation, over-transposition, or poor cell/nuclei quality — do not proceed with analysis
- **Nucleosomal banding**: Clear peaks at ~200 bp (mono), ~400 bp (di), and ~600 bp (tri) in the fragment size distribution confirm the Tn5 enzyme successfully accessed nucleosome-free regions. Absence of banding indicates incomplete transposition or degraded chromatin
- **Mitochondrial contamination**: chrM reads of 20–50% are normal for ATAC-seq (mitochondria have no nucleosomes and are highly accessible). Values > 60% indicate poor nuclear isolation — consider optimizing lysis conditions
- **FRiP score**: Requires called peaks (from Step 2) to compute. If FRiP < 0.2, try lowering the MACS2 p-value threshold or check whether the TSS enrichment score is adequate
- **Tn5 shift**: Skipping the +4/-5 bp correction will shift all peaks by ~4–5 bp. For most analyses this is negligible, but for precise footprinting (Step 3) it is essential

---

## Citation

If you use this skill in a publication, please cite:

- Corces, M.R. et al. (2017). An improved ATAC-seq protocol reduces background and enables interrogation of frozen tissues. *Nature Methods*, 14(10), 959–962.
- Buenrostro, J.D. et al. (2013). Transposition of native chromatin for fast and sensitive epigenomic profiling of open chromatin, DNA-binding proteins and nucleosome position. *Nature Methods*, 10(12), 1213–1218.
