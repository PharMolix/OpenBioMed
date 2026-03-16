# Raw Mass Spectrometry Data Processing (pyOpenMS)

Load, inspect, centroid, and extract features from raw LC-MS/MS data files.
This is **Step 1** of the proteomics pipeline — all downstream peptide
identification and quantification steps require centroided, quality-checked
spectra as input.

---

## What it does

1. Loads raw or profile-mode spectra from mzML, mzXML, or vendor-converted files using pyOpenMS
2. Inspects run-level QC metrics: total ion current (TIC), scan counts per MS level, m/z and RT ranges
3. Converts profile-mode spectra to centroid mode using PeakPickerHiRes
4. Extracts MS1 and MS2 spectra separately for downstream use
5. Detects LC-MS features (isotope envelopes) using FeatureFinder for label-free quantification
6. Extracts extracted ion chromatograms (EIC) for targeted m/z values
7. Converts between mzML, mzXML, and featureXML formats
8. Generates per-run QC plots (TIC, scan distribution, peak width)

---

## Why this exists

If you ask a general AI to "process my mzML files for proteomics," it will:

- Not distinguish between profile-mode and centroid-mode spectra (critical difference for downstream tools)
- Use incorrect pyOpenMS API calls (the API changed significantly between versions 2.x and 3.x)
- Skip quality control checks that reveal injection failures, column issues, or contamination
- Not explain the difference between peak picking and feature detection, or when each is needed
- Produce centroided output without verifying peak width or mass accuracy

This skill encodes the correct methodological decisions:

- Checks MS level distribution before processing to confirm DDA vs. DIA acquisition mode
- Applies PeakPickerHiRes (correct algorithm) not PeakPickerIterative (for Orbitrap data)
- Separates MS1 feature detection (for LFQ) from MS2 centroiding (for database search)
- Generates TIC plots to visually confirm run quality before investing compute time in search
- Uses pyOpenMS 3.x API (`MSExperiment`, `MzMLFile`) which differs from 2.x

---

## Reference Methods

**pyOpenMS** is the Python interface to OpenMS (Röst et al., 2016), a C++ framework for computational mass spectrometry. It provides direct access to the full OpenMS algorithm library including peak picking, feature detection, alignment, and quantification.

**PeakPickerHiRes:** Gaussian-based centroiding algorithm optimized for high-resolution instruments (Orbitrap, Q-TOF). Fits a Gaussian to each isotope peak and reports the apex m/z and intensity. Requires profile-mode input.

**FeatureFinderCentroided:** Identifies LC-MS features by grouping isotope envelopes across consecutive MS1 scans. Outputs feature maps with m/z, RT, intensity, and charge state per feature. Used as input to label-free quantification.

**File formats:**
- `mzML`: Open, XML-based vendor-neutral format. Use for all pipeline steps.
- `mzXML`: Older open format; convert to mzML with msconvert if needed.
- `featureXML`: OpenMS-native format for feature maps.
- `pepXML` / `mzIdentML`: Downstream peptide identification results formats.

---

## Usage (Python — pyOpenMS)

```python
import pyopenms as oms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Step 1: Load mzML file ────────────────────────────────────────────────────
exp = oms.MSExperiment()
oms.MzMLFile().load("sample.mzML", exp)

print(f"Spectra loaded:  {exp.getNrSpectra()}")
print(f"Chromatograms:   {exp.getNrChromatograms()}")

# ── Step 2: QC — inspect scan distribution ────────────────────────────────────
ms1_scans, ms2_scans = [], []
for spec in exp:
    level = spec.getMSLevel()
    if level == 1:
        ms1_scans.append(spec.getRT())
    elif level == 2:
        ms2_scans.append(spec.getRT())

print(f"MS1 scans: {len(ms1_scans)}")
print(f"MS2 scans: {len(ms2_scans)}")
print(f"MS2/MS1 ratio: {len(ms2_scans)/len(ms1_scans):.1f}  "
      f"(expect 5–20 for typical DDA)")

# ── Step 3: Plot Total Ion Chromatogram (TIC) ─────────────────────────────────
tic_rt, tic_int = [], []
for spec in exp:
    if spec.getMSLevel() == 1:
        tic_rt.append(spec.getRT() / 60)   # convert to minutes
        tic_int.append(spec.calculateTIC())

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(tic_rt, tic_int, lw=0.8, color="#2166AC")
ax.set_xlabel("Retention time (min)")
ax.set_ylabel("Total ion current")
ax.set_title("TIC — sample.mzML")
plt.tight_layout()
plt.savefig("figures/tic_sample.pdf")
plt.close()

# ── Step 4: Check acquisition mode (profile vs centroid) ─────────────────────
first_ms1 = next(s for s in exp if s.getMSLevel() == 1)
is_centroid = first_ms1.getType() == oms.SpectrumSettings.CENTROID
print(f"Acquisition mode: {'centroid' if is_centroid else 'profile'}")

# ── Step 5: Centroiding (only if profile-mode) ────────────────────────────────
if not is_centroid:
    picker = oms.PeakPickerHiRes()
    params = picker.getParameters()
    params.setValue("signal_to_noise", 1.0)    # lower = more peaks; 0.0 = all peaks
    params.setValue("ms_levels", [1, 2])
    picker.setParameters(params)

    centroided_exp = oms.MSExperiment()
    picker.pickExperiment(exp, centroided_exp, check_spectrum_type=False)

    print(f"Centroided spectra: {centroided_exp.getNrSpectra()}")
    oms.MzMLFile().store("sample_centroided.mzML", centroided_exp)
else:
    centroided_exp = exp
    print("Already centroided — no peak picking needed")

# ── Step 6: Extract a single MS2 spectrum ────────────────────────────────────
ms2_spectra = [s for s in centroided_exp if s.getMSLevel() == 2]
spec = ms2_spectra[0]

mz_array, int_array = spec.get_peaks()
precursor = spec.getPrecursors()[0]

print(f"\nExample MS2 spectrum:")
print(f"  Precursor m/z: {precursor.getMZ():.4f}")
print(f"  Precursor charge: {precursor.getCharge()}")
print(f"  Fragment peaks: {len(mz_array)}")
print(f"  RT: {spec.getRT()/60:.2f} min")

# ── Step 7: Feature detection (MS1, for LFQ) ──────────────────────────────────
ff = oms.FeatureFinder()
ff_name = "centroided"

features     = oms.FeatureMap()
seeds        = oms.FeatureMap()
ff_params    = oms.FeatureFinder().getParameters(ff_name)

# Extract MS1-only experiment for feature finding
ms1_exp = oms.MSExperiment()
for spec in centroided_exp:
    if spec.getMSLevel() == 1:
        ms1_exp.addSpectrum(spec)

ff.run(ff_name, ms1_exp, features, ff_params, seeds)
features.setUniqueIds()

print(f"\nFeatures detected: {features.size()}")
oms.FeatureXMLFile().store("sample_features.featureXML", features)

# ── Step 8: Export features as DataFrame ─────────────────────────────────────
rows = []
for feat in features:
    rows.append({
        "feature_id":  feat.getUniqueId(),
        "mz":          feat.getMZ(),
        "rt_min":      feat.getRT() / 60,
        "intensity":   feat.getIntensity(),
        "charge":      feat.getCharge(),
        "rt_start":    feat.getConvexHull().getBoundingBox().minX() / 60,
        "rt_end":      feat.getConvexHull().getBoundingBox().maxX() / 60,
    })

features_df = pd.DataFrame(rows)
features_df.to_csv("sample_features.csv", index=False)
print(features_df.head())

# ── Step 9: Extracted ion chromatogram (EIC) for a target m/z ────────────────
target_mz   = 524.2736    # example: target peptide precursor m/z
tolerance   = 0.02        # ± 20 mDa (Orbitrap); use 0.05 for Q-TOF

eic_rt, eic_int = [], []
for spec in centroided_exp:
    if spec.getMSLevel() == 1:
        mzs, ints = spec.get_peaks()
        mask = np.abs(mzs - target_mz) <= tolerance
        eic_rt.append(spec.getRT() / 60)
        eic_int.append(float(ints[mask].sum()) if mask.any() else 0.0)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(eic_rt, eic_int, lw=1.2, color="#E64B35")
ax.set_xlabel("Retention time (min)")
ax.set_ylabel("Intensity")
ax.set_title(f"EIC — m/z {target_mz:.4f} ± {tolerance*1000:.0f} mDa")
plt.tight_layout()
plt.savefig(f"figures/eic_{target_mz:.2f}.pdf")
plt.close()
```

## Batch processing multiple runs

```python
from pathlib import Path

mzml_files = list(Path("raw/").glob("*.mzML"))

for mzml_path in mzml_files:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)

    # QC summary
    ms1 = sum(1 for s in exp if s.getMSLevel() == 1)
    ms2 = sum(1 for s in exp if s.getMSLevel() == 2)
    print(f"{mzml_path.name}: {ms1} MS1, {ms2} MS2")

    # Centroid if needed
    first_ms1 = next(s for s in exp if s.getMSLevel() == 1)
    if first_ms1.getType() != oms.SpectrumSettings.CENTROID:
        picker = oms.PeakPickerHiRes()
        out_exp = oms.MSExperiment()
        picker.pickExperiment(exp, out_exp, check_spectrum_type=False)
        out_path = Path("centroided") / mzml_path.name
        oms.MzMLFile().store(str(out_path), out_exp)
        print(f"  → centroided: {out_path}")
```

---

## Example Output

```
Raw MS Data Processing
=======================
File: HeLa_digest_rep1.mzML
Spectra loaded:   18,432
Chromatograms:    0

Scan distribution:
  MS1 scans: 1,842
  MS2 scans: 16,590
  MS2/MS1 ratio: 9.0  (expect 5–20 for typical DDA)

Acquisition mode: profile
Centroiding with PeakPickerHiRes...
  Centroided spectra: 18,432  ✓

Feature detection (MS1):
  Features detected: 14,271
  Charge states: z=1 (12%), z=2 (38%), z=3 (31%), z=4+ (19%)
  RT range: 5.2 – 118.4 min
  m/z range: 300.1 – 1,650.4

Example MS2 spectrum (RT = 42.3 min):
  Precursor m/z: 524.2736
  Precursor charge: 2+
  Fragment peaks: 187

Exported:
  HeLa_digest_rep1_centroided.mzML
  HeLa_digest_rep1_features.featureXML
  HeLa_digest_rep1_features.csv   (14,271 features)
  figures/tic_HeLa_digest_rep1.pdf
```

---

## Interpretation Guide

- **MS2/MS1 ratio**: A ratio of 5–20 is typical for DDA experiments. Ratio < 3 suggests the instrument was triggering few MS2 events (check intensity threshold settings); ratio > 30 is unusual
- **Profile vs centroid**: Profile data shows Gaussian peak shapes per m/z; centroid data shows single m/z values per peak. Database search tools (MSFragger, Comet) require centroid input
- **TIC shape**: A broad, smooth Gaussian-like TIC indicates a good LC gradient. Flat TIC at the start/end is normal (column equilibration). Sudden drops mid-gradient indicate injection failure or column clogging
- **Feature count**: 10,000–30,000 features per run is typical for a HeLa cell digest on Orbitrap. Fewer than 5,000 suggests poor digest efficiency, short gradient, or injection problems
- **Charge state distribution**: z=2 and z=3 should dominate (> 60% combined) for a typical tryptic digest. High z=1 fraction suggests incomplete digestion or non-tryptic peptides
- **EIC tolerance**: Use ± 5–10 ppm for Orbitrap data (e.g., ± 0.003 Da at m/z 600); use ± 50 ppm or ± 0.05 Da for Q-TOF data

---

## Citation

If you use this skill in a publication, please cite:

- Röst, H.L. et al. (2016). OpenMS: a flexible open-source software platform for mass spectrometry data analysis. *Nature Methods*, 13(9), 741–748.
- Sturm, M. et al. (2008). OpenMS — an open-source software framework for mass spectrometry. *BMC Bioinformatics*, 9, 163.
