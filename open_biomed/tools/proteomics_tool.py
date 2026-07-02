"""
Single-cell proteomics data processing tool.
Load, inspect, centroid, and extract features from raw LC-MS/MS data files using pyOpenMS.

Supported operations:
- load: Load mzML/mzXML files and inspect QC metrics
- centroid: Convert profile-mode spectra to centroid mode
- feature_detection: Detect LC-MS features for label-free quantification
- eic: Extract extracted ion chromatograms for target m/z
- tic_plot: Generate TIC (Total Ion Chromatogram) plot
"""

import os
import uuid
import logging
from typing import Tuple, Dict, Any, Optional, Union, List

from open_biomed.tools.base_tool import Tool

logger = logging.getLogger('OpenBioMed')


class ProteomicsDataProcessing(Tool):
    """
    Process raw mass spectrometry data using pyOpenMS.

    This is Step 1 of the proteomics pipeline — provides centroided,
    quality-checked spectra as input for downstream peptide identification
    and quantification.

    Supported file formats:
    - mzML: Open, XML-based vendor-neutral format (recommended)
    - mzXML: Older open format; convert to mzML with msconvert if needed
    - featureXML: OpenMS-native format for feature maps

    Key operations:
    - load: Load and inspect raw data, QC metrics
    - centroid: Peak picking for high-resolution instruments (Orbitrap, Q-TOF)
    - feature_detection: Identify LC-MS features (isotope envelopes)
    - eic: Extract ion chromatogram for specific m/z
    - tic_plot: Generate QC visualization
    """

    def __init__(self) -> None:
        super().__init__()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import pyopenms as oms
            self._pyopenms_available = True
            self._oms = oms
        except ImportError as e:
            logger.warning(f"Missing pyopenms: {e}. Install with: pip install pyopenms")
            self._pyopenms_available = False
            self._oms = None

        try:
            import matplotlib.pyplot as plt
            self._matplotlib_available = True
            self._plt = plt
        except ImportError as e:
            logger.warning(f"Missing matplotlib: {e}. Install with: pip install matplotlib")
            self._matplotlib_available = False
            self._plt = None

        try:
            import numpy as np
            self._numpy_available = True
            self._np = np
        except ImportError as e:
            logger.warning(f"Missing numpy: {e}")
            self._numpy_available = False
            self._np = None

        try:
            import pandas as pd
            self._pandas_available = True
            self._pd = pd
        except ImportError as e:
            logger.warning(f"Missing pandas: {e}")
            self._pandas_available = False
            self._pd = None

    def print_usage(self) -> str:
        return """
Usage: Process raw mass spectrometry (LC-MS/MS) data using pyOpenMS
Inputs: {
    "file_path": str (path to mzML/mzXML file),
    "operation": str (load, centroid, feature_detection, eic, tic_plot),
    "output_dir": str (optional, default: ./tmp/),
    "target_mz": float (optional, for eic operation),
    "mz_tolerance": float (optional, m/z tolerance in Da, default: 0.02),
    "signal_to_noise": float (optional, S/N threshold for centroiding, default: 1.0)
}
Outputs: {
    "result": dict (operation-specific results),
    "message": str (status message)
}

Operations:
- load: Load file and return QC metrics (TIC, scan counts, m/z/RT ranges)
- centroid: Convert profile to centroid mode, save centroided mzML
- feature_detection: Detect MS1 features, save featureXML and CSV
- eic: Extract ion chromatogram for target m/z, save plot
- tic_plot: Generate TIC plot for QC visualization
"""

    def run(
        self,
        file_path: str,
        operation: str = "load",
        output_dir: str = "./tmp/",
        target_mz: Optional[float] = None,
        mz_tolerance: float = 0.02,
        signal_to_noise: float = 1.0,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Process mass spectrometry data.

        Args:
            file_path: Path to mzML/mzXML file
            operation: Operation to perform (load, centroid, feature_detection, eic, tic_plot)
            output_dir: Output directory for generated files
            target_mz: Target m/z for EIC extraction
            mz_tolerance: m/z tolerance in Da (default: 0.02, ~20 mDa for Orbitrap)
            signal_to_noise: S/N threshold for centroiding (default: 1.0)

        Returns:
            Tuple of (result_dict, message)
        """
        # Validate dependencies
        if not self._pyopenms_available:
            raise ImportError("pyopenms is required. Install with: pip install pyopenms")

        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique output ID
        output_id = str(uuid.uuid4())[:8]
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        logger.info(f"Processing {file_path} with operation: {operation}")

        # Load experiment first
        exp = self._oms.MSExperiment()
        if file_path.endswith('.mzML'):
            self._oms.MzMLFile().load(file_path, exp)
        elif file_path.endswith('.mzXML'):
            self._oms.MzXMLFile().load(file_path, exp)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Use mzML or mzXML.")

        # Execute operation
        if operation == "load":
            result = self._load_and_qc(exp, file_path)
        elif operation == "centroid":
            result = self._centroid(exp, base_name, output_dir, output_id, signal_to_noise)
        elif operation == "feature_detection":
            result = self._feature_detection(exp, base_name, output_dir, output_id)
        elif operation == "eic":
            if target_mz is None:
                raise ValueError("target_mz is required for eic operation")
            result = self._extract_eic(exp, base_name, output_dir, output_id, target_mz, mz_tolerance)
        elif operation == "tic_plot":
            result = self._generate_tic_plot(exp, base_name, output_dir, output_id)
        else:
            raise ValueError(f"Unknown operation: {operation}. Supported: load, centroid, feature_detection, eic, tic_plot")

        message = result.get("message", f"Operation {operation} completed successfully")
        return result, message

    def _load_and_qc(self, exp: Any, file_path: str) -> Dict[str, Any]:
        """Load and inspect QC metrics."""
        # Basic stats
        n_spectra = exp.getNrSpectra()
        n_chromatograms = exp.getNrChromatograms()

        # Scan distribution by MS level
        ms1_count, ms2_count = 0, 0
        ms1_rts, ms2_rts = [], []

        for spec in exp:
            level = spec.getMSLevel()
            if level == 1:
                ms1_count += 1
                ms1_rts.append(spec.getRT())
            elif level == 2:
                ms2_count += 1
                ms2_rts.append(spec.getRT())

        # Calculate MS2/MS1 ratio
        ms2_ms1_ratio = ms2_count / ms1_count if ms1_count > 0 else 0

        # Check acquisition mode (profile vs centroid)
        first_ms1 = next((s for s in exp if s.getMSLevel() == 1), None)
        is_centroid = False
        if first_ms1 is not None:
            is_centroid = first_ms1.getType() == self._oms.SpectrumSettings.CENTROID

        # Calculate TIC
        tic_values = []
        rt_values = []
        mz_range = [float('inf'), float('-inf')]
        rt_range = [float('inf'), float('-inf')]

        for spec in exp:
            if spec.getMSLevel() == 1:
                rt = spec.getRT()
                tic = spec.calculateTIC()
                rt_values.append(rt / 60)  # Convert to minutes
                tic_values.append(tic)

                # Update ranges
                mzs, _ = spec.get_peaks()
                if len(mzs) > 0:
                    mz_range[0] = min(mz_range[0], float(self._np.min(mzs)))
                    mz_range[1] = max(mz_range[1], float(self._np.max(mzs)))
                rt_range[0] = min(rt_range[0], rt / 60)
                rt_range[1] = max(rt_range[1], rt / 60)

        # Acquisition mode assessment
        acquisition_mode = "centroid" if is_centroid else "profile"
        dda_status = "typical DDA" if 5 <= ms2_ms1_ratio <= 20 else "check settings"

        return {
            "file_path": file_path,
            "n_spectra": n_spectra,
            "n_chromatograms": n_chromatograms,
            "ms1_count": ms1_count,
            "ms2_count": ms2_count,
            "ms2_ms1_ratio": round(ms2_ms1_ratio, 2),
            "acquisition_mode": acquisition_mode,
            "dda_status": dda_status,
            "mz_range": [round(mz_range[0], 4), round(mz_range[1], 4)] if mz_range[0] != float('inf') else None,
            "rt_range_min": [round(rt_range[0], 2), round(rt_range[1], 2)] if rt_range[0] != float('inf') else None,
            "tic_max": round(max(tic_values) if tic_values else 0, 2),
            "message": f"Loaded {n_spectra} spectra ({ms1_count} MS1, {ms2_count} MS2). Mode: {acquisition_mode}"
        }

    def _centroid(
        self,
        exp: Any,
        base_name: str,
        output_dir: str,
        output_id: str,
        signal_to_noise: float
    ) -> Dict[str, Any]:
        """Convert profile-mode spectra to centroid mode using PeakPickerHiRes."""

        # Check if already centroided
        first_ms1 = next((s for s in exp if s.getMSLevel() == 1), None)
        if first_ms1 is None:
            raise ValueError("No MS1 spectra found in file")

        is_centroid = first_ms1.getType() == self._oms.SpectrumSettings.CENTROID

        if is_centroid:
            # Already centroided, save as-is
            output_file = os.path.join(output_dir, f"{base_name}_centroided_{output_id}.mzML")
            self._oms.MzMLFile().store(output_file, exp)
            return {
                "output_file": output_file,
                "n_spectra": exp.getNrSpectra(),
                "already_centroided": True,
                "message": f"Already centroided. Saved to {output_file}"
            }

        # Perform centroiding with PeakPickerHiRes
        picker = self._oms.PeakPickerHiRes()
        params = picker.getParameters()
        params.setValue("signal_to_noise", signal_to_noise)
        params.setValue("ms_levels", [1, 2])
        picker.setParameters(params)

        centroided_exp = self._oms.MSExperiment()
        picker.pickExperiment(exp, centroided_exp, check_spectrum_type=False)

        # Save centroided file
        output_file = os.path.join(output_dir, f"{base_name}_centroided_{output_id}.mzML")
        self._oms.MzMLFile().store(output_file, centroided_exp)

        # QC check on centroided data
        ms1_count = sum(1 for s in centroided_exp if s.getMSLevel() == 1)
        ms2_count = sum(1 for s in centroided_exp if s.getMSLevel() == 2)

        return {
            "output_file": output_file,
            "n_spectra": centroided_exp.getNrSpectra(),
            "ms1_count": ms1_count,
            "ms2_count": ms2_count,
            "already_centroided": False,
            "signal_to_noise": signal_to_noise,
            "message": f"Centroided {centroided_exp.getNrSpectra()} spectra. Saved to {output_file}"
        }

    def _feature_detection(
        self,
        exp: Any,
        base_name: str,
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Detect LC-MS features (isotope envelopes) for label-free quantification."""

        # Ensure data is centroided
        first_ms1 = next((s for s in exp if s.getMSLevel() == 1), None)
        if first_ms1 is None:
            raise ValueError("No MS1 spectra found")

        is_centroid = first_ms1.getType() == self._oms.SpectrumSettings.CENTROID
        if not is_centroid:
            logger.warning("Data is not centroided. Feature detection may be suboptimal.")
            logger.info("Consider running 'centroid' operation first.")

        # Extract MS1-only experiment
        ms1_exp = self._oms.MSExperiment()
        for spec in exp:
            if spec.getMSLevel() == 1:
                ms1_exp.addSpectrum(spec)

        # Run feature finder
        ff = self._oms.FeatureFinder()
        ff_name = "centroided"

        features = self._oms.FeatureMap()
        seeds = self._oms.FeatureMap()
        ff_params = ff.getParameters(ff_name)

        ff.run(ff_name, ms1_exp, features, ff_params, seeds)
        features.setUniqueIds()

        # Save featureXML
        featurexml_file = os.path.join(output_dir, f"{base_name}_features_{output_id}.featureXML")
        self._oms.FeatureXMLFile().store(featurexml_file, features)

        # Export to CSV if pandas available
        csv_file = None
        if self._pandas_available:
            rows = []
            for feat in features:
                hull = feat.getConvexHull()
                bbox = hull.getBoundingBox() if hull.size() > 0 else None

                row = {
                    "feature_id": feat.getUniqueId(),
                    "mz": round(feat.getMZ(), 4),
                    "rt_min": round(feat.getRT() / 60, 2),
                    "intensity": round(feat.getIntensity(), 2),
                    "charge": feat.getCharge(),
                }
                if bbox is not None:
                    row["rt_start_min"] = round(bbox.minX() / 60, 2)
                    row["rt_end_min"] = round(bbox.maxX() / 60, 2)

                rows.append(row)

            features_df = self._pd.DataFrame(rows)
            csv_file = os.path.join(output_dir, f"{base_name}_features_{output_id}.csv")
            features_df.to_csv(csv_file, index=False)

        # Charge state distribution
        charge_dist = {}
        for feat in features:
            z = feat.getCharge()
            charge_dist[z] = charge_dist.get(z, 0) + 1

        return {
            "n_features": features.size(),
            "featurexml_file": featurexml_file,
            "csv_file": csv_file,
            "charge_distribution": charge_dist,
            "message": f"Detected {features.size()} features. Saved to {featurexml_file}"
        }

    def _extract_eic(
        self,
        exp: Any,
        base_name: str,
        output_dir: str,
        output_id: str,
        target_mz: float,
        mz_tolerance: float
    ) -> Dict[str, Any]:
        """Extract extracted ion chromatogram (EIC) for target m/z."""

        if not self._matplotlib_available or not self._numpy_available:
            raise ImportError("matplotlib and numpy are required for EIC extraction")

        eic_rt, eic_int = [], []

        for spec in exp:
            if spec.getMSLevel() == 1:
                mzs, ints = spec.get_peaks()
                mask = self._np.abs(mzs - target_mz) <= mz_tolerance
                eic_rt.append(spec.getRT() / 60)
                eic_int.append(float(ints[mask].sum()) if mask.any() else 0.0)

        # Generate plot
        fig, ax = self._plt.subplots(figsize=(8, 3))
        ax.plot(eic_rt, eic_int, lw=1.2, color="#E64B35")
        ax.set_xlabel("Retention time (min)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"EIC - m/z {target_mz:.4f} +/- {mz_tolerance*1000:.0f} mDa")

        plot_file = os.path.join(output_dir, f"{base_name}_eic_{target_mz:.2f}_{output_id}.pdf")
        self._plt.tight_layout()
        self._plt.savefig(plot_file)
        self._plt.close()

        # Find peak
        max_intensity = max(eic_int) if eic_int else 0
        peak_rt = eic_rt[eic_int.index(max_intensity)] if max_intensity > 0 else None

        return {
            "target_mz": target_mz,
            "mz_tolerance_da": mz_tolerance,
            "plot_file": plot_file,
            "max_intensity": round(max_intensity, 2),
            "peak_rt_min": round(peak_rt, 2) if peak_rt else None,
            "n_points": len(eic_rt),
            "message": f"EIC extracted for m/z {target_mz:.4f}. Peak at {peak_rt:.2f} min if peak_rt else 'no peak detected'"
        }

    def _generate_tic_plot(
        self,
        exp: Any,
        base_name: str,
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Generate Total Ion Chromatogram (TIC) plot for QC."""

        if not self._matplotlib_available:
            raise ImportError("matplotlib is required for TIC plot")

        tic_rt, tic_int = [], []

        for spec in exp:
            if spec.getMSLevel() == 1:
                tic_rt.append(spec.getRT() / 60)
                tic_int.append(spec.calculateTIC())

        # Generate plot
        fig, ax = self._plt.subplots(figsize=(10, 3))
        ax.plot(tic_rt, tic_int, lw=0.8, color="#2166AC")
        ax.set_xlabel("Retention time (min)")
        ax.set_ylabel("Total ion current")
        ax.set_title(f"TIC - {base_name}")

        plot_file = os.path.join(output_dir, f"{base_name}_tic_{output_id}.pdf")
        self._plt.tight_layout()
        self._plt.savefig(plot_file)
        self._plt.close()

        # TIC statistics
        max_tic = max(tic_int) if tic_int else 0
        max_rt = tic_rt[tic_int.index(max_tic)] if max_tic > 0 else None

        return {
            "plot_file": plot_file,
            "max_tic": round(max_tic, 2),
            "peak_rt_min": round(max_rt, 2) if max_rt else None,
            "rt_range_min": [round(min(tic_rt), 2), round(max(tic_rt), 2)] if tic_rt else None,
            "message": f"TIC plot generated. Saved to {plot_file}"
        }


if __name__ == "__main__":
    # Test the tool
    import sys

    if len(sys.argv) < 3:
        print("Usage: python proteomics_tool.py <file_path> <operation>")
        print("Operations: load, centroid, feature_detection, eic, tic_plot")
        print("For 'eic', also provide: --target_mz <value>")
        sys.exit(1)

    file_path = sys.argv[1]
    operation = sys.argv[2]

    kwargs = {}
    if operation == "eic" and len(sys.argv) >= 5:
        if sys.argv[3] == "--target_mz":
            kwargs["target_mz"] = float(sys.argv[4])

    tool = ProteomicsDataProcessing()
    result, message = tool.run(file_path, operation, **kwargs)
    print(f"Result: {result}")
    print(f"Message: {message}")