"""
Multi-omics data harmonization tool.
Prepare RNA-seq, proteomics, methylation, and other omics datasets for joint integration.

Supported operations:
- load: Load multi-omics CSV files into MuData container
- normalize: Apply per-data-type normalization
- batch_correct: ComBat batch correction using scanpy.pp.combat
- align_ids: Map UniProt/probe IDs to HGNC gene symbols via REST API
- impute: MinProb missing value imputation
- scale_export: Z-score scaling and export
- full_pipeline: Complete harmonization workflow
"""

import os
import uuid
import logging
import json
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List, Union
from collections import defaultdict

from open_biomed.tools.base_tool import Tool

logger = logging.getLogger('OpenBioMed')


class MultiOmicsHarmonization(Tool):
    """
    Harmonize multi-omics data for joint integration.

    Applies per-assay normalization, cross-assay batch correction,
    feature ID alignment, and missing value handling.

    Supported data types and normalization:
    - RNA-seq counts: normalize_total + log1p (scanpy)
    - Proteomics LFQ: log2 + median centering
    - Methylation β-values: M-value transformation (log2(β/(1-β)))
    - ATAC-seq peaks: log1p(CPM)
    - miRNA counts: log2(CPM + 1)

    Key features:
    - ComBat batch correction preserving biological signal
    - UniProt/probe ID → HGNC gene symbol mapping via Ensembl REST API
    - MinProb missing value imputation for proteomics
    - Z-score scaling for downstream integration
    """

    # Normalization methods per data type
    NORMALIZATION_METHODS = {
        "counts": "normalize_total_log1p",     # RNA-seq raw counts
        "lfq": "log2_median_centering",        # Proteomics LFQ intensity
        "beta": "m_value",                     # Methylation β-values
        "peak_counts": "log1p_cpm",            # ATAC-seq peaks
        "mirna_counts": "log2_cpm",            # miRNA counts
    }

    # Ensembl BioMart REST API endpoints
    ENSEMBL_BIOMART_URL = "https://rest.ensembl.org/biomart/martservice"

    def __init__(self) -> None:
        super().__init__()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import muon as mu
            self._muon_available = True
            self._mu = mu
        except ImportError as e:
            logger.warning(f"Missing muon: {e}. Install with: pip install muon")
            self._muon_available = False
            self._mu = None

        try:
            import scanpy as sc
            self._scanpy_available = True
            self._sc = sc
        except ImportError as e:
            logger.warning(f"Missing scanpy: {e}")
            self._scanpy_available = False
            self._sc = None

        try:
            import matplotlib.pyplot as plt
            self._matplotlib_available = True
            self._plt = plt
        except ImportError as e:
            logger.warning(f"Missing matplotlib: {e}")
            self._matplotlib_available = False
            self._plt = None

    def print_usage(self) -> str:
        return """
Usage: Harmonize multi-omics data for joint integration
Inputs: {
    "operation": str (load, normalize, batch_correct, align_ids, impute, scale_export, full_pipeline),
    "data_files": dict ({"rna": "path/to/rna.csv", "protein": "path/to/protein.csv"}),
    "sample_meta": str (path to sample_metadata.csv with Batch, Condition columns),
    "data_types": dict ({"rna": "counts", "protein": "lfq", "methylation": "beta"}),
    "batch_column": str (column name for batch, default: "Batch"),
    "condition_column": str (column name for condition, default: "Condition"),
    "missing_threshold": float (filter features with >X% missing, default: 0.30),
    "output_dir": str (output directory, default: ./tmp/),
    "export_format": str (h5mu, csv, or both, default: both)
}
Outputs: {
    "result": dict (operation-specific results),
    "message": str (status message)
}

Operations:
- load: Load CSV files into MuData container
- normalize: Apply per-data-type normalization
- batch_correct: ComBat batch correction
- align_ids: Map feature IDs to HGNC gene symbols
- impute: MinProb missing value imputation
- scale_export: Z-score and export harmonized data
- full_pipeline: Complete workflow

Normalization methods by data type:
- counts (RNA-seq): normalize_total + log1p
- lfq (proteomics): log2 + median centering
- beta (methylation): M-value transformation
- peak_counts (ATAC): log1p(CPM)
- mirna_counts: log2(CPM + 1)

Required dependencies:
- muon: pip install muon
- scanpy: pip install scanpy (includes sc.pp.combat)
"""

    def run(
        self,
        operation: str,
        data_files: Optional[Dict[str, str]] = None,
        sample_meta: Optional[str] = None,
        data_types: Optional[Dict[str, str]] = None,
        batch_column: str = "Batch",
        condition_column: str = "Condition",
        missing_threshold: float = 0.30,
        output_dir: str = "./tmp/",
        export_format: str = "both",
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Run multi-omics harmonization operation.

        Args:
            operation: Operation to perform
            data_files: Dict mapping assay name to CSV file path
            sample_meta: Path to sample metadata CSV (must have Batch, Condition columns)
            data_types: Dict mapping assay name to data type (counts, lfq, beta, etc.)
            batch_column: Column name for batch information
            condition_column: Column name for biological condition
            missing_threshold: Filter features with >X% missing values
            output_dir: Output directory
            export_format: Export format (h5mu, csv, or both)

        Returns:
            Tuple of (result_dict, message)
        """
        if not self._muon_available:
            raise ImportError("muon is required. Install with: pip install muon")

        if not self._scanpy_available:
            raise ImportError("scanpy is required. Install with: pip install scanpy")

        os.makedirs(output_dir, exist_ok=True)
        output_id = str(uuid.uuid4())[:8]

        logger.info(f"Running multi-omics harmonization operation: {operation}")

        # Store state for multi-step operations
        mdata = kwargs.get("mdata", None)
        harmonized = kwargs.get("harmonized", {})

        if operation == "load":
            if not data_files or not sample_meta:
                raise ValueError("data_files and sample_meta are required for load operation")
            result = self._load_data(
                data_files, sample_meta, data_types or {}, output_dir, output_id
            )
        elif operation == "normalize":
            if mdata is None:
                raise ValueError("mdata is required for normalize operation (run 'load' first)")
            result = self._normalize_data(
                mdata, data_types or {}, output_dir, output_id
            )
        elif operation == "batch_correct":
            if mdata is None:
                raise ValueError("mdata is required for batch_correct operation")
            result = self._batch_correct(
                mdata, batch_column, condition_column, output_dir, output_id
            )
        elif operation == "align_ids":
            if mdata is None:
                raise ValueError("mdata is required for align_ids operation")
            result = self._align_feature_ids(
                mdata, data_types or {}, output_dir, output_id
            )
        elif operation == "impute":
            if mdata is None:
                raise ValueError("mdata is required for impute operation")
            result = self._impute_missing(
                mdata, missing_threshold, output_dir, output_id
            )
        elif operation == "scale_export":
            if mdata is None:
                raise ValueError("mdata is required for scale_export operation")
            result = self._scale_and_export(
                mdata, output_dir, output_id, export_format
            )
        elif operation == "full_pipeline":
            if not data_files or not sample_meta:
                raise ValueError("data_files and sample_meta are required for full_pipeline")
            result = self._run_full_pipeline(
                data_files, sample_meta, data_types or {},
                batch_column, condition_column, missing_threshold,
                output_dir, output_id, export_format
            )
        else:
            raise ValueError(f"Unknown operation: {operation}. "
                           f"Supported: load, normalize, batch_correct, align_ids, impute, scale_export, full_pipeline")

        message = result.get("message", f"Operation {operation} completed")
        return result, message

    def _load_data(
        self,
        data_files: Dict[str, str],
        sample_meta: str,
        data_types: Dict[str, str],
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Load multi-omics CSV files into MuData container."""

        import anndata as ad
        from muon import MuData

        # Load sample metadata
        if not os.path.exists(sample_meta):
            raise FileNotFoundError(f"Sample metadata file not found: {sample_meta}")

        meta_df = pd.read_csv(sample_meta, index_col=0)

        # Check required columns
        required_cols = ["Batch", "Condition"]
        missing_cols = [c for c in required_cols if c not in meta_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in metadata: {missing_cols}. Will use defaults.")

        # Load each omics layer
        adatas = {}
        assay_info = {}

        for assay_name, file_path in data_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Load matrix (features x samples)
            mat_df = pd.read_csv(file_path, index_col=0)

            # Create AnnData (samples x features for scanpy convention)
            adata = ad.AnnData(
                X=mat_df.T.values,
                obs=meta_df.loc[mat_df.columns].copy() if set(mat_df.columns).issubset(meta_df.index) else pd.DataFrame(index=mat_df.columns),
                var=pd.DataFrame(index=mat_df.index),
                dtype=np.float32
            )

            # Store original feature names
            adata.var_names = mat_df.index.astype(str)

            # Store data type
            dtype = data_types.get(assay_name, "unknown")
            adata.uns["data_type"] = dtype

            adatas[assay_name] = adata
            assay_info[assay_name] = {
                "n_features": mat_df.shape[0],
                "n_samples": mat_df.shape[1],
                "data_type": dtype,
                "file": file_path
            }

        # Create MuData
        mdata = MuData(adatas)

        # Intersect samples across all assays
        self._mu.pp.intersect_obs(mdata)

        # Save MuData
        mdata_file = os.path.join(output_dir, f"raw_mudata_{output_id}.h5mu")
        mdata.write(mdata_file)

        n_common_samples = mdata.n_obs

        return {
            "status": "success",
            "mdata_file": mdata_file,
            "n_assays": len(adatas),
            "assays": assay_info,
            "n_common_samples": n_common_samples,
            "batch_info": meta_df.get("Batch", pd.Series()).value_counts().to_dict() if "Batch" in meta_df.columns else {},
            "mdata": mdata,  # Pass to next operation
            "message": f"Loaded {len(adatas)} assays with {n_common_samples} common samples"
        }

    def _normalize_data(
        self,
        mdata: Any,
        data_types: Dict[str, str],
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Apply per-data-type normalization."""

        import scanpy as sc

        normalization_summary = {}

        for assay_name in mdata.mod.keys():
            adata = mdata.mod[assay_name]
            dtype = adata.uns.get("data_type", data_types.get(assay_name, "unknown"))

            logger.info(f"Normalizing {assay_name} ({dtype})...")

            if dtype == "counts":
                # RNA-seq: normalize_total + log1p (approximation to VST)
                sc.pp.normalize_total(adata, target_sum=1e6)
                sc.pp.log1p(adata)
                normalization_summary[assay_name] = "normalize_total(1e6) + log1p"

            elif dtype == "lfq":
                # Proteomics: log2 + median centering
                X = adata.X.copy()
                X = np.log2(X + 1)  # log2(x+1) to handle zeros
                # Median centering per sample
                sample_medians = np.median(X, axis=1, keepdims=True)
                global_median = np.median(X)
                X = X - sample_medians + global_median
                adata.X = X.astype(np.float32)
                normalization_summary[assay_name] = "log2 + median centering"

            elif dtype == "beta":
                # Methylation: M-value transformation
                X = adata.X.copy()
                # Clamp to avoid division by zero
                X = np.clip(X, 0.001, 0.999)
                # M = log2(beta / (1 - beta))
                X = np.log2(X / (1 - X))
                adata.X = X.astype(np.float32)
                normalization_summary[assay_name] = "M-value transformation"

            elif dtype == "peak_counts":
                # ATAC-seq: log1p(CPM)
                X = adata.X.copy()
                # CPM normalization
                col_sums = X.sum(axis=1, keepdims=True)
                X = X / col_sums * 1e6
                X = np.log1p(X)
                adata.X = X.astype(np.float32)
                normalization_summary[assay_name] = "log1p(CPM)"

            elif dtype == "mirna_counts":
                # miRNA: log2(CPM + 1)
                X = adata.X.copy()
                col_sums = X.sum(axis=1, keepdims=True)
                X = X / col_sums * 1e6
                X = np.log2(X + 1)
                adata.X = X.astype(np.float32)
                normalization_summary[assay_name] = "log2(CPM + 1)"

            else:
                # Default: log1p
                sc.pp.log1p(adata)
                normalization_summary[assay_name] = "log1p (default)"

            # Store normalization method
            adata.uns["normalization"] = normalization_summary[assay_name]

        # Save normalized MuData
        norm_file = os.path.join(output_dir, f"normalized_mudata_{output_id}.h5mu")
        mdata.write(norm_file)

        return {
            "status": "success",
            "mdata_file": norm_file,
            "normalization_summary": normalization_summary,
            "mdata": mdata,
            "message": f"Normalized {len(normalization_summary)} assays"
        }

    def _batch_correct(
        self,
        mdata: Any,
        batch_column: str,
        condition_column: str,
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Apply ComBat batch correction using scanpy.pp.combat."""

        import scanpy as sc

        # Check batch-confounding
        batch_correction_summary = {}
        pca_results = {}

        for assay_name in mdata.mod.keys():
            adata = mdata.mod[assay_name]

            # Check if batch column exists
            if batch_column not in adata.obs.columns:
                logger.warning(f"No batch column '{batch_column}' in {assay_name}, skipping batch correction")
                continue

            # Check batch-condition confounding
            if condition_column in adata.obs.columns:
                confound_table = pd.crosstab(adata.obs[batch_column], adata.obs[condition_column])
                logger.info(f"Batch-Condition table for {assay_name}:\n{confound_table}")

                # If completely confounded (each batch has only one condition), skip
                if confound_table.apply(lambda x: (x > 0).sum() == 1, axis=1).all():
                    logger.warning(f"Batch and Condition are confounded in {assay_name}, skipping batch correction")
                    batch_correction_summary[assay_name] = "skipped (batch-condition confounded)"
                    continue

            # Generate PCA before correction
            if self._matplotlib_available:
                try:
                    sc.tl.pca(adata)
                    # Store PC coordinates for comparison
                    pca_results[f"{assay_name}_before"] = {
                        "PC1": adata.obsm["X_pca"][:, 0].tolist()[:10],
                        "PC2": adata.obsm["X_pca"][:, 1].tolist()[:10]
                    }
                except Exception as e:
                    logger.warning(f"PCA before correction failed for {assay_name}: {e}")

            # Apply ComBat
            try:
                # Use condition as covariate to preserve biological signal
                covariates = [condition_column] if condition_column in adata.obs.columns else None
                sc.pp.combat(adata, key=batch_column, covariates=covariates)
                batch_correction_summary[assay_name] = f"ComBat (batch={batch_column}, covariates={covariates})"
            except Exception as e:
                logger.warning(f"ComBat failed for {assay_name}: {e}")
                batch_correction_summary[assay_name] = f"failed: {str(e)}"
                continue

            # Generate PCA after correction
            if self._matplotlib_available:
                try:
                    sc.tl.pca(adata)
                    pca_results[f"{assay_name}_after"] = {
                        "PC1": adata.obsm["X_pca"][:, 0].tolist()[:10],
                        "PC2": adata.obsm["X_pca"][:, 1].tolist()[:10]
                    }
                except Exception as e:
                    logger.warning(f"PCA after correction failed for {assay_name}: {e}")

        # Save corrected MuData
        corrected_file = os.path.join(output_dir, f"corrected_mudata_{output_id}.h5mu")
        mdata.write(corrected_file)

        return {
            "status": "success",
            "mdata_file": corrected_file,
            "batch_correction_summary": batch_correction_summary,
            "pca_comparison": pca_results,
            "mdata": mdata,
            "message": f"Batch correction applied to {len([k for k, v in batch_correction_summary.items() if 'ComBat' in v])} assays"
        }

    def _align_feature_ids(
        self,
        mdata: Any,
        data_types: Dict[str, str],
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Map UniProt/probe IDs to HGNC gene symbols via Ensembl REST API."""

        id_mapping_summary = {}

        for assay_name in mdata.mod.keys():
            adata = mdata.mod[assay_name]
            dtype = adata.uns.get("data_type", data_types.get(assay_name, "unknown"))

            if dtype == "lfq":
                # Proteomics: UniProt → HGNC
                uniprot_ids = adata.var_names.tolist()
                mapped_ids = self._map_uniprot_to_hgnc(uniprot_ids)

                # Update var_names
                new_var_names = [mapped_ids.get(uid, uid) for uid in uniprot_ids]
                adata.var_names = new_var_names
                adata.var["original_id"] = uniprot_ids

                n_mapped = len([k for k, v in mapped_ids.items() if v != k])
                id_mapping_summary[assay_name] = f"UniProt→HGNC: {n_mapped}/{len(uniprot_ids)} mapped"

            elif dtype == "beta":
                # Methylation: Illumina probe → HGNC (simplified - use probe ID as is)
                # Full mapping requires Illumina annotation file
                id_mapping_summary[assay_name] = "probe IDs retained (requires Illumina annotation for mapping)"

            else:
                id_mapping_summary[assay_name] = "no ID mapping needed"

        # Save aligned MuData
        aligned_file = os.path.join(output_dir, f"aligned_mudata_{output_id}.h5mu")
        mdata.write(aligned_file)

        return {
            "status": "success",
            "mdata_file": aligned_file,
            "id_mapping_summary": id_mapping_summary,
            "mdata": mdata,
            "message": f"Feature ID alignment completed"
        }

    def _map_uniprot_to_hgnc(self, uniprot_ids: List[str]) -> Dict[str, str]:
        """Map UniProt accessions to HGNC gene symbols via Ensembl BioMart REST API."""

        mapped = {}
        batch_size = 100

        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i:i + batch_size]

            try:
                # Ensembl BioMart query
                query_xml = f"""
                <Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1">
                    <Dataset name="hsapiens_gene_ensembl" interface="default">
                        <Filter name="uniprotswissprot_accession" value="{','.join(batch)}"/>
                        <Attribute name="uniprotswissprot_accession"/>
                        <Attribute name="hgnc_symbol"/>
                    </Dataset>
                </Query>
                """

                response = requests.post(
                    self.ENSEMBL_BIOMART_URL,
                    data={"query": query_xml},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30
                )

                if response.status_code == 200:
                    for line in response.text.strip().split("\n"):
                        if line:
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                uniprot_id, hgnc_symbol = parts[0], parts[1]
                                if hgnc_symbol:
                                    mapped[uniprot_id] = hgnc_symbol

            except Exception as e:
                logger.warning(f"BioMart query failed for batch {i}: {e}")

        return mapped

    def _impute_missing(
        self,
        mdata: Any,
        missing_threshold: float,
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """Filter high-missingness features and impute remaining with MinProb."""

        imputation_summary = {}

        for assay_name in mdata.mod.keys():
            adata = mdata.mod[assay_name]

            # Calculate missingness per feature
            X = adata.X
            if isinstance(X, np.ndarray):
                missing_rate = np.isnan(X).mean(axis=0)  # per feature
            else:
                # Sparse matrix
                missing_rate = np.zeros(adata.n_vars)  # assume no missing in sparse

            # Filter high missingness
            keep_features = missing_rate < missing_threshold
            n_filtered = (~keep_features).sum()
            adata._inplace_subset_var(keep_features)

            # MinProb imputation for remaining missing values
            dtype = adata.uns.get("data_type", "unknown")

            if dtype == "lfq":
                # Proteomics MNAR: impute from low-intensity distribution
                X = adata.X.copy()
                for j in range(X.shape[1]):  # per sample
                    missing_mask = np.isnan(X[:, j])
                    if missing_mask.any():
                        # Sample from distribution centered at 1st percentile
                        q01 = np.nanquantile(X[:, j], 0.01)
                        std = abs(q01) * 0.1
                        X[missing_mask, j] = np.random.normal(q01, std, missing_mask.sum())
                adata.X = X.astype(np.float32)
                imputation_summary[assay_name] = f"MinProb: {n_filtered} filtered, {missing_rate[keep_features].mean()*100:.1f}% imputed"
            else:
                imputation_summary[assay_name] = f"{n_filtered} features filtered (> {missing_threshold*100:.0f}% missing)"

        # Save imputed MuData
        imputed_file = os.path.join(output_dir, f"imputed_mudata_{output_id}.h5mu")
        mdata.write(imputed_file)

        return {
            "status": "success",
            "mdata_file": imputed_file,
            "imputation_summary": imputation_summary,
            "mdata": mdata,
            "message": f"Missing value handling completed"
        }

    def _scale_and_export(
        self,
        mdata: Any,
        output_dir: str,
        output_id: str,
        export_format: str
    ) -> Dict[str, Any]:
        """Z-score scaling and export harmonized data."""

        export_files = {}

        for assay_name in mdata.mod.keys():
            adata = mdata.mod[assay_name]

            # Z-score scaling (per feature)
            X = adata.X.copy()
            means = X.mean(axis=0, keepdims=True)
            stds = X.std(axis=0, keepdims=True)
            stds[stds == 0] = 1  # avoid division by zero
            X = (X - means) / stds
            adata.X = X.astype(np.float32)

        # Export
        if export_format in ["h5mu", "both"]:
            h5mu_file = os.path.join(output_dir, f"harmonized_multiomics_{output_id}.h5mu")
            mdata.write(h5mu_file)
            export_files["h5mu"] = h5mu_file

        if export_format in ["csv", "both"]:
            # Export each assay as CSV
            for assay_name in mdata.mod.keys():
                adata = mdata.mod[assay_name]
                csv_file = os.path.join(output_dir, f"harmonized_{assay_name}_{output_id}.csv")
                # features x samples
                df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
                df.to_csv(csv_file)
                export_files[f"{assay_name}_csv"] = csv_file

        # Summary
        summary = {
            assay_name: {
                "n_features": mdata.mod[assay_name].n_vars,
                "n_samples": mdata.mod[assay_name].n_obs,
                "normalization": mdata.mod[assay_name].uns.get("normalization", "unknown")
            }
            for assay_name in mdata.mod.keys()
        }

        return {
            "status": "success",
            "export_files": export_files,
            "assay_summary": summary,
            "n_samples": mdata.n_obs,
            "n_assays": len(mdata.mod),
            "message": f"Harmonized data exported: {len(export_files)} files"
        }

    def _run_full_pipeline(
        self,
        data_files: Dict[str, str],
        sample_meta: str,
        data_types: Dict[str, str],
        batch_column: str,
        condition_column: str,
        missing_threshold: float,
        output_dir: str,
        output_id: str,
        export_format: str
    ) -> Dict[str, Any]:
        """Run complete harmonization pipeline."""

        # Step 1: Load
        load_result = self._load_data(data_files, sample_meta, data_types, output_dir, output_id)
        mdata = load_result["mdata"]

        # Step 2: Normalize
        norm_result = self._normalize_data(mdata, data_types, output_dir, output_id)
        mdata = norm_result["mdata"]

        # Step 3: Batch correct
        batch_result = self._batch_correct(mdata, batch_column, condition_column, output_dir, output_id)
        mdata = batch_result["mdata"]

        # Step 4: Align IDs
        align_result = self._align_feature_ids(mdata, data_types, output_dir, output_id)
        mdata = align_result["mdata"]

        # Step 5: Impute
        impute_result = self._impute_missing(mdata, missing_threshold, output_dir, output_id)
        mdata = impute_result["mdata"]

        # Step 6: Scale and export
        export_result = self._scale_and_export(mdata, output_dir, output_id, export_format)

        return {
            "status": "success",
            "load": {"n_assays": load_result["n_assays"], "n_samples": load_result["n_common_samples"]},
            "normalization": norm_result["normalization_summary"],
            "batch_correction": batch_result["batch_correction_summary"],
            "id_mapping": align_result["id_mapping_summary"],
            "imputation": impute_result["imputation_summary"],
            "export_files": export_result["export_files"],
            "assay_summary": export_result["assay_summary"],
            "message": f"Full pipeline completed. {len(export_result['export_files'])} files exported"
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(MultiOmicsHarmonization().print_usage())
        sys.exit(1)

    tool = MultiOmicsHarmonization()
    print(tool.print_usage())