"""
Single-cell RNA-seq analysis tool using Scanpy.
Complete workflow for scRNA-seq data analysis including QC, normalization,
dimensionality reduction, clustering, and marker gene identification.

Supported operations:
- load: Load h5ad, h5 (10X), mtx, or CSV files
- qc: Quality control and filtering
- normalize: Normalization, log-transformation, and HVG selection
- cluster: PCA, UMAP, and Leiden clustering
- markers: Marker gene identification with statistical testing
"""

import os
import uuid
import logging
from typing import Tuple, Dict, Any, Optional, Union, List

from open_biomed.tools.base_tool import Tool

logger = logging.getLogger('OpenBioMed')


class ScanpyAnalysis(Tool):
    """
    Complete single-cell RNA-seq analysis workflow using Scanpy.

    This tool provides a Step-by-step pipeline for scRNA-seq analysis:
    1. Load data from various formats (h5ad, 10X h5, mtx, CSV)
    2. Quality control (mitochondrial gene detection, cell/gene filtering)
    3. Normalization (total-count normalization, log-transformation)
    4. Highly variable gene selection
    5. Dimensionality reduction (PCA, UMAP/t-SNE)
    6. Clustering (Leiden algorithm)
    7. Marker gene identification (Wilcoxon test)

    Supported file formats:
    - h5ad: AnnData native format (recommended)
    - h5: 10X Genomics HDF5 format
    - mtx: 10X Genomics MTX format (with genes.tsv, barcodes.tsv)
    - csv: Expression matrix as CSV
    """

    def __init__(self) -> None:
        super().__init__()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import scanpy as sc
            import anndata as ad
            self._scanpy_available = True
            self._sc = sc
            self._ad = ad
            # Configure scanpy settings
            sc.settings.verbosity = 2
            sc.settings.set_figure_params(dpi=80, facecolor='white')
        except ImportError as e:
            logger.warning(f"Missing scanpy/anndata: {e}. Install with: pip install scanpy anndata")
            self._scanpy_available = False
            self._sc = None
            self._ad = None

        try:
            import matplotlib.pyplot as plt
            self._matplotlib_available = True
            self._plt = plt
        except ImportError as e:
            logger.warning(f"Missing matplotlib: {e}")
            self._matplotlib_available = False
            self._plt = None

        try:
            import pandas as pd
            self._pandas_available = True
            self._pd = pd
        except ImportError as e:
            logger.warning(f"Missing pandas: {e}")
            self._pandas_available = False
            self._pd = None

        try:
            import numpy as np
            self._numpy_available = True
            self._np = np
        except ImportError as e:
            logger.warning(f"Missing numpy: {e}")
            self._numpy_available = False
            self._np = None

    def print_usage(self) -> str:
        return """
Usage: Single-cell RNA-seq analysis using Scanpy
Inputs: {
    "file_path": str (path to h5ad, h5, mtx directory, or csv file),
    "operation": str (load, qc, normalize, cluster, markers, full_pipeline),
    "output_dir": str (optional, default: ./tmp/),
    "min_genes": int (optional, min genes per cell for QC, default: 200),
    "min_cells": int (optional, min cells per gene for QC, default: 3),
    "max_mt_percent": float (optional, max mitochondrial percentage, default: 5.0),
    "n_top_genes": int (optional, number of HVGs, default: 2000),
    "n_neighbors": int (optional, neighbors for graph, default: 10),
    "n_pcs": int (optional, PCs for neighborhood, default: 40),
    "resolution": float (optional, Leiden resolution, default: 0.5),
    "groupby": str (optional, groupby for markers, default: leiden)
}
Outputs: {
    "result": dict (operation-specific results including file paths and metrics),
    "message": str (status message)
}

Operations:
- load: Load data file and return basic statistics
- qc: Calculate QC metrics and filter cells/genes
- normalize: Normalize counts, log-transform, select HVGs
- cluster: PCA, neighborhood graph, UMAP, Leiden clustering
- markers: Identify marker genes per cluster
- full_pipeline: Complete workflow from load to markers
"""

    def run(
        self,
        file_path: str,
        operation: str = "load",
        output_dir: str = "./tmp/",
        min_genes: int = 200,
        min_cells: int = 3,
        max_mt_percent: float = 5.0,
        n_top_genes: int = 2000,
        n_neighbors: int = 10,
        n_pcs: int = 40,
        resolution: float = 0.5,
        groupby: str = "leiden",
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Run single-cell RNA-seq analysis.

        Args:
            file_path: Path to data file (h5ad, h5, mtx directory, or csv)
            operation: Operation to perform (load, qc, normalize, cluster, markers, full_pipeline)
            output_dir: Output directory for generated files
            min_genes: Minimum genes per cell for QC filtering
            min_cells: Minimum cells per gene for QC filtering
            max_mt_percent: Maximum mitochondrial gene percentage
            n_top_genes: Number of highly variable genes to select
            n_neighbors: Number of neighbors for graph construction
            n_pcs: Number of PCs for neighborhood graph
            resolution: Leiden clustering resolution
            groupby: Groupby key for marker gene identification

        Returns:
            Tuple of (result_dict, message)
        """
        # Validate dependencies
        if not self._scanpy_available:
            raise ImportError("scanpy and anndata are required. Install with: pip install scanpy anndata")

        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

        # Generate unique output ID
        output_id = str(uuid.uuid4())[:8]
        self._sc.settings.figdir = os.path.join(output_dir, "figures/")

        logger.info(f"Processing {file_path} with operation: {operation}")

        # Execute operation
        if operation == "load":
            result = self._load_data(file_path, output_dir, output_id)
        elif operation == "qc":
            result = self._quality_control(file_path, output_dir, output_id, min_genes, min_cells, max_mt_percent)
        elif operation == "normalize":
            adata = self._load_data(file_path, output_dir, output_id, return_adata=True)
            result = self._normalize_and_hvg(adata, output_dir, output_id, n_top_genes)
        elif operation == "cluster":
            adata = self._load_data(file_path, output_dir, output_id, return_adata=True)
            result = self._clustering(adata, output_dir, output_id, n_neighbors, n_pcs, resolution)
        elif operation == "markers":
            adata = self._load_data(file_path, output_dir, output_id, return_adata=True)
            result = self._marker_genes(adata, output_dir, output_id, groupby)
        elif operation == "full_pipeline":
            result = self._full_pipeline(
                file_path, output_dir, output_id,
                min_genes, min_cells, max_mt_percent,
                n_top_genes, n_neighbors, n_pcs, resolution
            )
        else:
            raise ValueError(f"Unknown operation: {operation}. Supported: load, qc, normalize, cluster, markers, full_pipeline")

        message = result.get("message", f"Operation {operation} completed successfully")
        return result, message

    def _load_data(
        self,
        file_path: str,
        output_dir: str,
        output_id: str,
        return_adata: bool = False
    ) -> Union[Dict[str, Any], Any]:
        """Load data from various formats."""
        adata = None
        format_type = None

        # Detect format and load
        if file_path.endswith('.h5ad'):
            adata = self._sc.read_h5ad(file_path)
            format_type = "h5ad"
        elif file_path.endswith('.h5'):
            adata = self._sc.read_10x_h5(file_path)
            format_type = "10x_h5"
        elif os.path.isdir(file_path):
            # Check for mtx files in directory
            if os.path.exists(os.path.join(file_path, 'matrix.mtx')):
                adata = self._sc.read_10x_mtx(file_path)
                format_type = "10x_mtx"
            else:
                raise ValueError(f"MTX directory must contain matrix.mtx file: {file_path}")
        elif file_path.endswith('.csv'):
            adata = self._sc.read_csv(file_path)
            format_type = "csv"
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Use h5ad, h5, mtx directory, or csv.")

        # Basic statistics
        n_obs = adata.n_obs
        n_vars = adata.n_vars

        # Check for existing preprocessing
        has_raw = adata.raw is not None
        has_pca = 'X_pca' in adata.obsm
        has_umap = 'X_umap' in adata.obsm
        has_clusters = 'leiden' in adata.obs.columns

        if return_adata:
            return adata

        # Save loaded data
        output_file = os.path.join(output_dir, f"loaded_{output_id}.h5ad")
        adata.write_h5ad(output_file)

        return {
            "output_file": output_file,
            "format": format_type,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "has_raw": has_raw,
            "has_pca": has_pca,
            "has_umap": has_umap,
            "has_clusters": has_clusters,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "message": f"Loaded {format_type} data: {n_obs} cells x {n_vars} genes. Saved to {output_file}"
        }

    def _quality_control(
        self,
        file_path: str,
        output_dir: str,
        output_id: str,
        min_genes: int,
        min_cells: int,
        max_mt_percent: float
    ) -> Dict[str, Any]:
        """Calculate QC metrics and filter cells/genes."""
        adata = self._load_data(file_path, output_dir, output_id, return_adata=True)

        # Identify mitochondrial genes (common patterns: MT- for human, mt- for mouse)
        adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')

        # Calculate QC metrics
        self._sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # Store pre-filter stats
        pre_filter_obs = adata.n_obs
        pre_filter_vars = adata.n_vars

        # Generate QC plots if matplotlib available
        if self._matplotlib_available:
            qc_plot_file = os.path.join(output_dir, "figures", f"qc_metrics_{output_id}.pdf")
            self._sc.pl.violin(
                adata,
                ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4,
                multi_panel=True,
                save=f"_qc_metrics_{output_id}.pdf"
            )
            # Rename to expected path
            if os.path.exists(os.path.join(output_dir, "figures", f"violin_qc_metrics_{output_id}.pdf")):
                os.rename(
                    os.path.join(output_dir, "figures", f"violin_qc_metrics_{output_id}.pdf"),
                    qc_plot_file
                )

        # Filter cells by min_genes
        self._sc.pp.filter_cells(adata, min_genes=min_genes)

        # Filter genes by min_cells
        self._sc.pp.filter_genes(adata, min_cells=min_cells)

        # Filter by mitochondrial percentage
        if max_mt_percent < 100:
            adata = adata[adata.obs.pct_counts_mt < max_mt_percent, :]

        # Post-filter stats
        post_filter_obs = adata.n_obs
        post_filter_vars = adata.n_vars
        cells_removed = pre_filter_obs - post_filter_obs
        genes_removed = pre_filter_vars - post_filter_vars

        # Save filtered data
        output_file = os.path.join(output_dir, f"filtered_{output_id}.h5ad")
        adata.write_h5ad(output_file)

        return {
            "output_file": output_file,
            "pre_filter_cells": pre_filter_obs,
            "pre_filter_genes": pre_filter_vars,
            "post_filter_cells": post_filter_obs,
            "post_filter_genes": post_filter_vars,
            "cells_removed": cells_removed,
            "genes_removed": genes_removed,
            "min_genes": min_genes,
            "min_cells": min_cells,
            "max_mt_percent": max_mt_percent,
            "mt_genes_detected": int(adata.var['mt'].sum()),
            "qc_plot": qc_plot_file if self._matplotlib_available else None,
            "message": f"QC completed. Filtered {cells_removed} cells, {genes_removed} genes. Saved to {output_file}"
        }

    def _normalize_and_hvg(
        self,
        adata: Any,
        output_dir: str,
        output_id: str,
        n_top_genes: int
    ) -> Dict[str, Any]:
        """Normalize counts, log-transform, and select highly variable genes."""
        # Check if already normalized
        if adata.raw is not None:
            logger.info("Raw counts already backed up. Proceeding with normalization...")

        # Total-count normalization (CPM-like, targeting 10,000 counts per cell)
        self._sc.pp.normalize_total(adata, target_sum=1e4)

        # Log-transformation
        self._sc.pp.log1p(adata)

        # Store raw counts for downstream use
        if adata.raw is None:
            adata.raw = adata

        # Identify highly variable genes
        # Use 'seurat' flavor instead of 'seurat_v3' as it's more robust for normalized data
        try:
            self._sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        except Exception as e:
            logger.warning(f"seurat flavor failed: {e}, falling back to cell_ranger")
            self._sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')

        # HVG statistics
        n_hvgs = int(adata.var['highly_variable'].sum())
        mean_hvg_dispersion = float(adata.var.loc[adata.var['highly_variable'], 'dispersions'].mean())

        # Generate HVG plot if matplotlib available
        if self._matplotlib_available:
            hvg_plot_file = os.path.join(output_dir, "figures", f"hvg_{output_id}.pdf")
            self._sc.pl.highly_variable_genes(adata, save=f"_hvg_{output_id}.pdf")
            if os.path.exists(os.path.join(output_dir, "figures", f"highly_variable_genes_hvg_{output_id}.pdf")):
                os.rename(
                    os.path.join(output_dir, "figures", f"highly_variable_genes_hvg_{output_id}.pdf"),
                    hvg_plot_file
                )

        # Save normalized data
        output_file = os.path.join(output_dir, f"normalized_{output_id}.h5ad")
        adata.write_h5ad(output_file)

        return {
            "output_file": output_file,
            "n_top_genes": n_top_genes,
            "n_hvgs_detected": n_hvgs,
            "mean_hvg_dispersion": round(mean_hvg_dispersion, 4),
            "target_sum": 10000,
            "hvg_plot": hvg_plot_file if self._matplotlib_available else None,
            "message": f"Normalization completed. {n_hvgs} HVGs selected. Saved to {output_file}"
        }

    def _clustering(
        self,
        adata: Any,
        output_dir: str,
        output_id: str,
        n_neighbors: int,
        n_pcs: int,
        resolution: float
    ) -> Dict[str, Any]:
        """PCA, neighborhood graph, UMAP, and Leiden clustering."""
        # Ensure HVGs are selected
        if 'highly_variable' not in adata.var.columns:
            logger.warning("No HVGs found. Selecting HVGs automatically...")
            self._sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')

        # Subset to HVGs for clustering
        if adata.var['highly_variable'].any():
            adata = adata[:, adata.var['highly_variable']]

        # Scale data (regress out total_counts and pct_counts_mt if available)
        if 'total_counts' in adata.obs.columns and 'pct_counts_mt' in adata.obs.columns:
            self._sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
        else:
            logger.info("Skipping regression (QC metrics not found)")

        self._sc.pp.scale(adata, max_value=10)

        # PCA
        self._sc.tl.pca(adata, svd_solver='arpack')

        # PCA variance ratio
        if self._matplotlib_available:
            pca_plot_file = os.path.join(output_dir, "figures", f"pca_variance_{output_id}.pdf")
            self._sc.pl.pca_variance_ratio(adata, log=True, n_pcs=min(50, adata.n_vars), save=f"_pca_{output_id}.pdf")
            if os.path.exists(os.path.join(output_dir, "figures", f"pca_variance_ratio_pca_{output_id}.pdf")):
                os.rename(
                    os.path.join(output_dir, "figures", f"pca_variance_ratio_pca_{output_id}.pdf"),
                    pca_plot_file
                )

        # Neighborhood graph
        self._sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        # UMAP
        self._sc.tl.umap(adata)

        # Leiden clustering
        self._sc.tl.leiden(adata, resolution=resolution)

        # Cluster statistics
        n_clusters = len(adata.obs['leiden'].unique())
        cluster_sizes = adata.obs['leiden'].value_counts().to_dict()

        # UMAP plot with clusters
        if self._matplotlib_available:
            umap_plot_file = os.path.join(output_dir, "figures", f"umap_clusters_{output_id}.pdf")
            self._sc.pl.umap(adata, color='leiden', legend_loc='on data', save=f"_clusters_{output_id}.pdf")
            if os.path.exists(os.path.join(output_dir, "figures", f"umap_clusters_{output_id}.pdf")):
                # Already named correctly
                umap_plot_file = os.path.join(output_dir, "figures", f"umap_clusters_{output_id}.pdf")

        # Save clustered data
        output_file = os.path.join(output_dir, f"clustered_{output_id}.h5ad")
        adata.write_h5ad(output_file)

        return {
            "output_file": output_file,
            "n_clusters": n_clusters,
            "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "resolution": resolution,
            "umap_plot": umap_plot_file if self._matplotlib_available else None,
            "pca_plot": pca_plot_file if self._matplotlib_available else None,
            "message": f"Clustering completed. Found {n_clusters} clusters. Saved to {output_file}"
        }

    def _marker_genes(
        self,
        adata: Any,
        output_dir: str,
        output_id: str,
        groupby: str
    ) -> Dict[str, Any]:
        """Identify marker genes for each cluster."""
        # Check if groupby column exists
        if groupby not in adata.obs.columns:
            raise ValueError(f"Groupby column '{groupby}' not found in adata.obs. Available: {list(adata.obs.columns)}")

        # Run marker gene identification (Wilcoxon rank-sum test)
        self._sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon')

        # Get marker results as DataFrame
        markers_df = self._sc.get.rank_genes_groups_df(adata, group=None)

        # Export markers to CSV
        markers_csv = os.path.join(output_dir, f"markers_{output_id}.csv")
        if self._pandas_available:
            markers_df.to_csv(markers_csv, index=False)

        # Generate marker plots
        if self._matplotlib_available:
            # Heatmap of top markers
            heatmap_plot = os.path.join(output_dir, "figures", f"markers_heatmap_{output_id}.pdf")
            self._sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby=groupby, save=f"_heatmap_{output_id}.pdf")
            if os.path.exists(os.path.join(output_dir, "figures", f"heatmap_markers_heatmap_{output_id}.pdf")):
                os.rename(
                    os.path.join(output_dir, "figures", f"heatmap_markers_heatmap_{output_id}.pdf"),
                    heatmap_plot
                )

            # Dot plot
            dotplot_plot = os.path.join(output_dir, "figures", f"markers_dotplot_{output_id}.pdf")
            self._sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, groupby=groupby, save=f"_dotplot_{output_id}.pdf")
            if os.path.exists(os.path.join(output_dir, "figures", f"dotplot_markers_dotplot_{output_id}.pdf")):
                os.rename(
                    os.path.join(output_dir, "figures", f"dotplot_markers_dotplot_{output_id}.pdf"),
                    dotplot_plot
                )

        # Summarize top markers per cluster
        top_markers = {}
        n_groups = len(adata.obs[groupby].unique())

        if self._pandas_available:
            for group in adata.obs[groupby].unique():
                group_markers = markers_df[markers_df['group'] == str(group)]
                top_genes = group_markers.head(10)['names'].tolist()
                top_scores = group_markers.head(10)['scores'].tolist()
                top_markers[str(group)] = {
                    "genes": top_genes,
                    "scores": [round(s, 4) for s in top_scores]
                }

        # Save data with markers
        output_file = os.path.join(output_dir, f"with_markers_{output_id}.h5ad")
        adata.write_h5ad(output_file)

        return {
            "output_file": output_file,
            "markers_csv": markers_csv,
            "n_groups": n_groups,
            "groupby": groupby,
            "method": "wilcoxon",
            "top_markers": top_markers,
            "heatmap_plot": heatmap_plot if self._matplotlib_available else None,
            "dotplot_plot": dotplot_plot if self._matplotlib_available else None,
            "message": f"Marker genes identified for {n_groups} groups. Saved to {markers_csv}"
        }

    def _full_pipeline(
        self,
        file_path: str,
        output_dir: str,
        output_id: str,
        min_genes: int,
        min_cells: int,
        max_mt_percent: float,
        n_top_genes: int,
        n_neighbors: int,
        n_pcs: int,
        resolution: float
    ) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        logger.info("Running full scRNA-seq analysis pipeline...")

        # Step 1: Load
        adata = self._load_data(file_path, output_dir, output_id, return_adata=True)
        load_stats = {
            "n_obs_initial": adata.n_obs,
            "n_vars_initial": adata.n_vars
        }

        # Step 2: QC
        qc_result = self._quality_control(file_path, output_dir, output_id, min_genes, min_cells, max_mt_percent)
        adata = self._sc.read_h5ad(qc_result["output_file"])

        # Step 3: Normalize
        norm_result = self._normalize_and_hvg(adata, output_dir, output_id, n_top_genes)

        # Step 4: Cluster (need to reload normalized data)
        adata = self._sc.read_h5ad(norm_result["output_file"])
        cluster_result = self._clustering(adata, output_dir, output_id, n_neighbors, n_pcs, resolution)

        # Step 5: Markers (need to reload clustered data)
        adata = self._sc.read_h5ad(cluster_result["output_file"])
        markers_result = self._marker_genes(adata, output_dir, output_id, "leiden")

        # Final output file
        final_output = os.path.join(output_dir, f"pipeline_complete_{output_id}.h5ad")
        adata.write_h5ad(final_output)

        # Compile all results
        return {
            "output_file": final_output,
            "pipeline_steps": ["load", "qc", "normalize", "cluster", "markers"],
            "load_stats": load_stats,
            "qc_stats": {
                "cells_removed": qc_result["cells_removed"],
                "genes_removed": qc_result["genes_removed"],
                "post_filter_cells": qc_result["post_filter_cells"]
            },
            "normalize_stats": {
                "n_hvgs": norm_result["n_hvgs_detected"]
            },
            "cluster_stats": {
                "n_clusters": cluster_result["n_clusters"],
                "resolution": resolution
            },
            "markers_stats": {
                "n_groups": markers_result["n_groups"],
                "markers_csv": markers_result["markers_csv"]
            },
            "figures": {
                "qc_plot": qc_result.get("qc_plot"),
                "hvg_plot": norm_result.get("hvg_plot"),
                "pca_plot": cluster_result.get("pca_plot"),
                "umap_plot": cluster_result.get("umap_plot"),
                "heatmap_plot": markers_result.get("heatmap_plot"),
                "dotplot_plot": markers_result.get("dotplot_plot")
            },
            "message": f"Full pipeline completed. {cluster_result['n_clusters']} clusters found. Final data saved to {final_output}"
        }


if __name__ == "__main__":
    # Test the tool
    import sys

    if len(sys.argv) < 3:
        print("Usage: python scanpy_analysis_tool.py <file_path> <operation>")
        print("Operations: load, qc, normalize, cluster, markers, full_pipeline")
        print("Optional: --min_genes 200 --min_cells 3 --max_mt_percent 5.0 --resolution 0.5")
        sys.exit(1)

    file_path = sys.argv[1]
    operation = sys.argv[2]

    kwargs = {}
    for i in range(3, len(sys.argv), 2):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]
            value = sys.argv[i+1]
            # Convert to appropriate type
            if key in ['min_genes', 'min_cells', 'n_top_genes', 'n_neighbors', 'n_pcs']:
                kwargs[key] = int(value)
            elif key in ['max_mt_percent', 'resolution']:
                kwargs[key] = float(value)
            else:
                kwargs[key] = value

    tool = ScanpyAnalysis()
    result, message = tool.run(file_path, operation, **kwargs)
    print(f"Result: {result}")
    print(f"Message: {message}")