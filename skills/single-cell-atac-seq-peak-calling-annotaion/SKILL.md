# ATAC-seq Peak Calling and Differential Accessibility

Call accessible chromatin peaks from ATAC-seq BAM files, annotate peaks to
genomic features and genes, and identify differentially accessible regions
between experimental conditions.

This is **Step 2** of the bulk ATAC-seq pipeline.

---

## What it does

1. Calls peaks with MACS2 (`--nomodel --shift -100 --extsize 200`) optimized for ATAC-seq
2. Filters peaks overlapping ENCODE blacklist regions
3. Builds a consensus peak set across all samples using bedtools merge
4. Counts reads per sample in consensus peaks (featureCounts or bedtools coverage)
5. Annotates peaks with genomic features (promoter, UTR, exon, intron, intergenic) and nearest gene
6. Runs differential accessibility analysis with DESeq2 on pseudobulk counts (recommended) or DiffBind
7. Applies FDR correction and generates ranked DAR table and volcano plot

---

## Why this exists

If you ask a general AI to "call peaks from ATAC-seq data," it will:

- Use default MACS2 parameters (designed for ChIP-seq), not ATAC-seq-specific parameters
- Not build a consensus peak set across samples, making cross-sample comparison impossible
- Not remove ENCODE blacklist regions, leaving artifactual peaks from repetitive elements
- Use `--format BAM` instead of `--format BAMPE` (ATAC-seq is paired-end)
- Apply DESeq2 directly to individual reads per cell instead of pseudobulk aggregation for multi-sample data

This skill encodes the correct methodological decisions:

- Uses ATAC-seq-specific MACS2 flags: `--nomodel --shift -100 --extsize 200 --format BAMPE`
- Filters ENCODE blacklist regions (hg38 or mm10) that produce artifactual signal
- Builds a reproducible consensus peak set using IDR or bedtools merge across replicates
- Applies pseudobulk DESeq2 for multi-sample differential analysis (controls type I error)

---

## Reference Methods

**MACS2 ATAC-seq parameters:**
- `--nomodel`: Skip the ChIP enrichment model building (ATAC-seq does not have a broad enrichment model)
- `--shift -100 --extsize 200`: Centers signal on Tn5 cut site by shifting reads 100 bp upstream and extending 200 bp
- `--format BAMPE`: Reads paired-end fragment coordinates directly from BAM
- `--nolambda`: Disables local lambda background estimation (optional; use with high-coverage data)

**Consensus peak set:** Union of peaks called across all samples, merged within 100 bp. This fixed peak universe ensures every sample contributes a count for every region, enabling proper statistical comparison.

**Peak annotation categories** (distance from peak center to nearest TSS):
- Promoter: ≤ 3 kb upstream of TSS
- 5' UTR / 3' UTR
- Exon / Intron
- Distal intergenic (> 3 kb from any gene)

**Pseudobulk DESeq2:** Aggregate fragment counts per condition replicate, then apply DESeq2. Avoids pseudo-replication that occurs when treating individual cells or fragments as independent observations.

---

## Usage

### Peak calling (Bash)

```bash
#!/bin/bash
GENOME_SIZE="hs"   # "hs" for human, "mm" for mouse
BLACKLIST="hg38-blacklist.v2.bed"
THREADS=8

for BAM in aligned/*_final.bam; do
  SAMPLE=$(basename ${BAM} _final.bam)

  # ── Call peaks with MACS2 (ATAC-seq mode) ───────────────────────────────────
  macs2 callpeak \
    -t ${BAM} \
    -f BAMPE \
    -n ${SAMPLE} \
    --outdir peaks/ \
    -g ${GENOME_SIZE} \
    --nomodel \
    --shift -100 \
    --extsize 200 \
    --keep-dup all \
    --call-summits \
    -p 0.01 \
    2> logs/${SAMPLE}_macs2.log

  # ── Filter blacklist regions ─────────────────────────────────────────────────
  bedtools intersect \
    -v \
    -a peaks/${SAMPLE}_peaks.narrowPeak \
    -b ${BLACKLIST} \
  > peaks/${SAMPLE}_peaks_filtered.narrowPeak

  echo "${SAMPLE}: $(wc -l < peaks/${SAMPLE}_peaks_filtered.narrowPeak) peaks after blacklist filter"
done

# ── Build consensus peak set ──────────────────────────────────────────────────
# Concatenate all filtered peaks, sort, and merge within 100 bp
cat peaks/*_peaks_filtered.narrowPeak \
| awk '{OFS="\t"; print $1, $2, $3}' \
| sort -k1,1 -k2,2n \
| bedtools merge -i stdin -d 100 \
> consensus_peaks.bed

echo "Consensus peaks: $(wc -l < consensus_peaks.bed)"

# ── Count reads per sample per consensus peak ─────────────────────────────────
for BAM in aligned/*_final.bam; do
  SAMPLE=$(basename ${BAM} _final.bam)
  bedtools coverage \
    -a consensus_peaks.bed \
    -b ${BAM} \
    -counts \
  | awk '{print $4}' \
  > counts/${SAMPLE}_counts.txt
done

# Combine into count matrix
paste consensus_peaks.bed counts/*_counts.txt > count_matrix.tsv
echo "Count matrix: $(wc -l < count_matrix.tsv) peaks × $(ls counts/*.txt | wc -l) samples"
```

### Peak annotation (R — ChIPseeker)

```r
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(org.Hs.eg.db)
library(ggplot2)

txdb    <- TxDb.Hsapiens.UCSC.hg38.knownGene
peaks   <- readPeakFile("consensus_peaks.bed", as = "GRanges")

# Annotate peaks
anno <- annotatePeak(
  peaks,
  tssRegion    = c(-3000, 3000),
  TxDb         = txdb,
  annoDb       = "org.Hs.eg.db",
  addFlankGeneInfo = TRUE
)

# Visualize annotation distribution
plotAnnoPie(anno, main = "Peak genomic distribution")
plotDistToTSS(anno, main = "Distance to nearest TSS")

# Export annotated peak table
anno_df <- as.data.frame(anno)
write.csv(anno_df, "annotated_peaks.csv", row.names = FALSE)

cat("Promoter peaks:", sum(grepl("Promoter", anno_df$annotation)), "\n")
cat("Intergenic peaks:", sum(grepl("Intergenic", anno_df$annotation)), "\n")
```

### Differential accessibility (R — DESeq2 pseudobulk)

```r
library(DESeq2)
library(ggplot2)
library(ggrepel)

# Load count matrix
counts_raw <- read.table("count_matrix.tsv", header = FALSE)
sample_meta <- read.csv("sample_metadata.csv")

peak_ids    <- paste0(counts_raw$V1, ":", counts_raw$V2, "-", counts_raw$V3)
count_mat   <- as.matrix(counts_raw[, 4:ncol(counts_raw)])
rownames(count_mat) <- peak_ids
colnames(count_mat) <- sample_meta$SampleID

# DESeq2
dds <- DESeqDataSetFromMatrix(
  countData = count_mat,
  colData   = sample_meta,
  design    = ~ Condition
)

# Filter low-count peaks (min 10 reads in at least 3 samples)
keep <- rowSums(counts(dds) >= 10) >= 3
dds  <- dds[keep, ]
cat("Peaks after filtering:", nrow(dds), "\n")

dds <- DESeq(dds)

# Extract results
res <- results(dds,
               contrast  = c("Condition", "Disease", "Control"),
               alpha     = 0.05,
               lfcThreshold = 0)
res <- lfcShrink(dds, contrast = c("Condition","Disease","Control"),
                  res = res, type = "ashr")

res_df <- as.data.frame(res)
res_df$peak_id <- rownames(res_df)
res_df <- res_df[order(res_df$padj), ]

# Annotate with nearest gene
res_annot <- merge(res_df,
                   anno_df[, c("V4","SYMBOL","annotation")],
                   by.x = "peak_id", by.y = "V4", all.x = TRUE)

write.csv(res_annot, "dar_results.csv", row.names = FALSE)

# Summary
sig <- res_annot[!is.na(res_annot$padj) &
                   res_annot$padj < 0.05 &
                   abs(res_annot$log2FoldChange) > 1, ]
cat(sprintf("Significant DARs (|log2FC|>1, FDR<5%%): %d\n", nrow(sig)))
cat(sprintf("  More accessible in Disease: %d\n", sum(sig$log2FoldChange > 0)))
cat(sprintf("  Less accessible in Disease: %d\n", sum(sig$log2FoldChange < 0)))

# Volcano plot
plot_df <- res_annot[!is.na(res_annot$padj), ]
plot_df$sig <- ifelse(plot_df$padj < 0.05 & abs(plot_df$log2FoldChange) > 1,
                      ifelse(plot_df$log2FoldChange > 0, "Open", "Closed"), "NS")

ggplot(plot_df, aes(log2FoldChange, -log10(padj), color = sig)) +
  geom_point(size = 0.8, alpha = 0.6) +
  scale_color_manual(values = c(Open="#E64B35", Closed="#4DBBD5", NS="#AAAAAA")) +
  geom_hline(yintercept = -log10(0.05), lty=2, color="grey50") +
  geom_vline(xintercept = c(-1, 1), lty=2, color="grey50") +
  geom_text_repel(data = head(sig, 20),
                   aes(label = SYMBOL), size=2.5, max.overlaps=15) +
  labs(x = "log2 Fold Change (Disease / Control)",
       y = "-log10 FDR",
       title = "Differential Chromatin Accessibility") +
  theme_bw(base_size=12) +
  theme(legend.title=element_blank())
ggsave("figures/dar_volcano.pdf", width=7, height=6)
```

---

## Example Output

```
ATAC-seq Peak Calling and Differential Accessibility
======================================================
Samples: 6  (3 Disease, 3 Control)

Peak calling (MACS2, p < 0.01):
  Disease_rep1:  52,341 peaks  (after blacklist filter)
  Disease_rep2:  49,873 peaks
  Control_rep1:  54,102 peaks
  Control_rep2:  51,847 peaks

Consensus peak set:  72,431 peaks (bedtools merge, d=100 bp)

Peak annotation (hg38):
  Promoter (≤3kb):   18,421  (25.4%)
  5'/3' UTR:          2,847   (3.9%)
  Exon:               4,112   (5.7%)
  Intron:            22,341  (30.8%)
  Intergenic:        24,710  (34.1%)

Differential accessibility (DESeq2, Disease vs Control):
  Peaks tested: 68,312 (after low-count filter)
  Significant DARs (|log2FC|>1, FDR<5%): 4,821
    More accessible in Disease: 2,614
    Less accessible in Disease: 2,207

Top open DARs in Disease:
  chr2:87,324,112-87,325,008  MKI67 promoter  log2FC=+3.2, FDR=2.1e-12
  chr17:7,687,445-7,688,112   TP53 promoter   log2FC=+2.8, FDR=4.3e-11

Exported:
  annotated_peaks.csv   (72,431 peaks)
  dar_results.csv       (68,312 peaks with statistics)
  figures/dar_volcano.pdf
```

---

## Interpretation Guide

- **Consensus peak count**: 50,000–150,000 consensus peaks is typical for bulk ATAC-seq across human cell types. Fewer than 20,000 suggests low library complexity or failed peak calling; more than 300,000 may indicate too permissive p-value threshold
- **Promoter vs. intergenic ratio**: For most cell types, 25–35% of accessible peaks are at promoters; 30–40% are distal intergenic (potential enhancers). Cell-type-specific enhancers are the most biologically interesting distal peaks
- **DAR FDR threshold**: 5% FDR and |log2FC| > 1 is standard. For transcription factor binding sites (narrow peaks), the fold change can be very high (4–8×); for broad histone marks it may be smaller (1–2×)
- **IDR vs. bedtools merge for consensus peaks**: IDR (Irreproducibility Discovery Rate) is more stringent and requires exactly 2 replicates per condition. For ≥ 3 replicates or multi-condition designs, bedtools merge is more practical
- **p-value vs. q-value for MACS2**: Use `-p 0.01` (p-value) for peak calling; the q-value default (`-q 0.05`) is more conservative and may miss real peaks in lower-quality samples

---

## Citation

If you use this skill in a publication, please cite:

- Zhang, Y. et al. (2008). Model-based Analysis of ChIP-Seq (MACS). *Genome Biology*, 9(9), R137. (MACS2)
- Yu, G. et al. (2015). ChIPseeker: an R/Bioconductor package for ChIP peak annotation, comparison and visualization. *Bioinformatics*, 31(14), 2382–2383.
- Love, M.I. et al. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15, 550.
