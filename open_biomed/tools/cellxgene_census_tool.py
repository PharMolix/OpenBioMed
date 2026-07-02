"""
CELLxGENE Census query tool.
Query CZ CELLxGENE Census (61M+ cells) for single-cell expression data.

Supported operations:
- get_summary: Get census summary statistics
- get_datasets: List available datasets
- get_obs: Query cell metadata
- get_var: Query gene metadata
- get_anndata: Retrieve expression data as AnnData
"""

import os
import uuid
import logging
from typing import Tuple, Dict, Any, Optional, Union, List

from open_biomed.tools.base_tool import Tool

logger = logging.getLogger('OpenBioMed')


class CellxGeneCensusQuery(Tool):
    """
    Query CZ CELLxGENE Census for single-cell expression data.

    This tool provides access to the CZ CELLxGENE Census:
    - 61+ million cells from human and mouse
    - Standardized metadata (cell types, tissues, diseases, donors)
    - Raw gene expression matrices
    - Pre-calculated embeddings and statistics

    Key operations:
    - get_summary: Get census version and cell counts
    - get_datasets: List datasets with metadata
    - get_obs: Query cell metadata by filters
    - get_var: Query gene metadata
    - get_anndata: Retrieve expression data as AnnData

    Important: Always filter for is_primary_data == True to avoid duplicate cells.
    """

    def __init__(self) -> None:
        super().__init__()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import cellxgene_census
            self._cellxgene_available = True
            self._census = cellxgene_census
        except ImportError as e:
            logger.warning(f"Missing cellxgene-census: {e}. Install with: pip install cellxgene-census")
            self._cellxgene_available = False
            self._census = None

        try:
            import pandas as pd
            self._pandas_available = True
            self._pd = pd
        except ImportError as e:
            logger.warning(f"Missing pandas: {e}")
            self._pandas_available = False
            self._pd = None

        try:
            import anndata as ad
            self._anndata_available = True
            self._ad = ad
        except ImportError as e:
            logger.warning(f"Missing anndata: {e}")
            self._anndata_available = False
            self._ad = None

    def print_usage(self) -> str:
        return """
Usage: Query CZ CELLxGENE Census for single-cell expression data
Inputs: {
    "operation": str (get_summary, get_datasets, get_obs, get_var, get_anndata),
    "organism": str (homo_sapiens, mus_musculus, default: homo_sapiens),
    "obs_value_filter": str (optional, cell filter e.g., "cell_type == 'B cell' and is_primary_data == True"),
    "var_value_filter": str (optional, gene filter e.g., "feature_name in ['CD4', 'CD8A']"),
    "obs_column_names": list (optional, metadata columns to retrieve),
    "var_column_names": list (optional, gene metadata columns),
    "census_version": str (optional, default: "stable"),
    "output_dir": str (optional, default: ./tmp/),
    "max_cells": int (optional, max cells for get_anndata, default: 100000)
}
Outputs: {
    "result": dict (operation-specific results),
    "message": str (status message)
}

Operations:
- get_summary: Get census version and total cell counts
- get_datasets: List available datasets with metadata
- get_obs: Query cell metadata (returns DataFrame info)
- get_var: Query gene metadata (returns DataFrame info)
- get_anndata: Retrieve expression data as AnnData (saved to h5ad)

Important: Always use "is_primary_data == True" filter to avoid duplicate cells.
"""

    def run(
        self,
        operation: str = "get_summary",
        organism: str = "homo_sapiens",
        obs_value_filter: Optional[str] = None,
        var_value_filter: Optional[str] = None,
        obs_column_names: Optional[List[str]] = None,
        var_column_names: Optional[List[str]] = None,
        census_version: str = "stable",
        output_dir: str = "./tmp/",
        max_cells: int = 100000,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Query CELLxGENE Census.

        Args:
            operation: Operation to perform (get_summary, get_datasets, get_obs, get_var, get_anndata)
            organism: Organism name (homo_sapiens, mus_musculus)
            obs_value_filter: Filter for cells (e.g., "cell_type == 'B cell' and is_primary_data == True")
            var_value_filter: Filter for genes (e.g., "feature_name in ['CD4', 'CD8A']")
            obs_column_names: Metadata columns to retrieve for cells
            var_column_names: Metadata columns to retrieve for genes
            census_version: Census version (default: "stable")
            output_dir: Output directory for saved files
            max_cells: Maximum cells to retrieve for get_anndata

        Returns:
            Tuple of (result_dict, message)
        """
        # Validate dependencies
        if not self._cellxgene_available:
            raise ImportError("cellxgene-census is required. Install with: pip install cellxgene-census")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique output ID
        output_id = str(uuid.uuid4())[:8]

        logger.info(f"Querying CELLxGENE Census: operation={operation}, organism={organism}")

        # Execute operation
        if operation == "get_summary":
            result = self._get_summary(census_version)
        elif operation == "get_datasets":
            result = self._get_datasets(census_version, organism, output_dir, output_id)
        elif operation == "get_obs":
            result = self._get_obs(organism, obs_value_filter, obs_column_names, census_version)
        elif operation == "get_var":
            result = self._get_var(organism, var_value_filter, var_column_names, census_version)
        elif operation == "get_anndata":
            result = self._get_anndata(
                organism, obs_value_filter, var_value_filter,
                obs_column_names, var_column_names,
                census_version, output_dir, output_id, max_cells
            )
        else:
            raise ValueError(f"Unknown operation: {operation}. Supported: get_summary, get_datasets, get_obs, get_var, get_anndata")

        message = result.get("message", f"Operation {operation} completed successfully")
        return result, message

    def _get_summary(self, census_version: str) -> Dict[str, Any]:
        """Get census summary statistics."""
        with self._census.open_soma(census_version=census_version) as census:
            # Get summary info
            summary_df = census["census_info"]["summary"].read().concat().to_pandas()

            # Extract key statistics
            total_cells = int(summary_df['total_cell_count'].iloc[0]) if 'total_cell_count' in summary_df.columns else 0
            census_version_actual = census_version

            # Get organism-specific counts if available
            organism_counts = {}
            try:
                for organism in ["homo_sapiens", "mus_musculus"]:
                    if f"census_data[{organism}]" in summary_df.columns:
                        organism_counts[organism] = int(summary_df[f"census_data[{organism}]"].iloc[0])
            except Exception:
                pass

            return {
                "census_version": census_version_actual,
                "total_cell_count": total_cells,
                "organism_counts": organism_counts,
                "summary_data": summary_df.to_dict(orient="records") if self._pandas_available else [],
                "message": f"Census version {census_version_actual} contains {total_cells} total cells"
            }

    def _get_datasets(
        self,
        census_version: str,
        organism: str,
        output_dir: str,
        output_id: str
    ) -> Dict[str, Any]:
        """List available datasets."""
        with self._census.open_soma(census_version=census_version) as census:
            # Get datasets info
            datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()

            # Filter by organism if specified
            if organism and "organism" in datasets_df.columns:
                datasets_df = datasets_df[datasets_df["organism"] == organism]

            n_datasets = len(datasets_df)

            # Save to CSV
            csv_file = None
            if self._pandas_available and n_datasets > 0:
                csv_file = os.path.join(output_dir, f"census_datasets_{output_id}.csv")
                datasets_df.to_csv(csv_file, index=False)

            # Get summary statistics
            dataset_summary = {
                "n_datasets": n_datasets,
                "n_cells_total": int(datasets_df["dataset_total_cell_count"].sum()) if "dataset_total_cell_count" in datasets_df.columns else 0,
                "unique_tissues": len(datasets_df["tissue"].unique()) if "tissue" in datasets_df.columns else 0,
                "unique_diseases": len(datasets_df["disease"].unique()) if "disease" in datasets_df.columns else 0,
            }

            return {
                "n_datasets": n_datasets,
                "csv_file": csv_file,
                "summary": dataset_summary,
                "columns": list(datasets_df.columns),
                "message": f"Found {n_datasets} datasets for {organism}. Saved to {csv_file if csv_file else 'N/A'}"
            }

    def _get_obs(
        self,
        organism: str,
        obs_value_filter: Optional[str],
        obs_column_names: Optional[List[str]],
        census_version: str
    ) -> Dict[str, Any]:
        """Query cell metadata."""
        # Default filter to avoid duplicates
        if obs_value_filter is None:
            obs_value_filter = "is_primary_data == True"

        # Add is_primary_data filter if not present
        if "is_primary_data" not in obs_value_filter:
            obs_value_filter = f"{obs_value_filter} and is_primary_data == True"

        # Default columns
        if obs_column_names is None:
            obs_column_names = ["cell_type", "tissue_general", "disease", "donor_id", "sex", "assay"]

        with self._census.open_soma(census_version=census_version) as census:
            obs_df = self._census.get_obs(
                census,
                organism,
                value_filter=obs_value_filter,
                column_names=obs_column_names
            )

            n_cells = len(obs_df)

            # Get unique value counts for key columns
            unique_counts = {}
            for col in ["cell_type", "tissue_general", "disease"]:
                if col in obs_df.columns:
                    unique_counts[col] = int(obs_df[col].nunique())

            # Sample of unique values
            sample_values = {}
            for col in ["cell_type", "tissue_general"]:
                if col in obs_df.columns:
                    sample_values[col] = list(obs_df[col].unique()[:10])

            return {
                "n_cells": n_cells,
                "organism": organism,
                "obs_value_filter": obs_value_filter,
                "columns": list(obs_df.columns),
                "unique_counts": unique_counts,
                "sample_values": sample_values,
                "message": f"Found {n_cells} cells matching filter: {obs_value_filter}"
            }

    def _get_var(
        self,
        organism: str,
        var_value_filter: Optional[str],
        var_column_names: Optional[List[str]],
        census_version: str
    ) -> Dict[str, Any]:
        """Query gene metadata."""
        # Default columns
        if var_column_names is None:
            var_column_names = ["feature_id", "feature_name", "feature_length"]

        with self._census.open_soma(census_version=census_version) as census:
            var_df = self._census.get_var(
                census,
                organism,
                value_filter=var_value_filter,
                column_names=var_column_names
            )

            n_genes = len(var_df)

            # Get gene names if available
            gene_names = []
            if "feature_name" in var_df.columns:
                gene_names = list(var_df["feature_name"].unique()[:20])

            return {
                "n_genes": n_genes,
                "organism": organism,
                "var_value_filter": var_value_filter,
                "columns": list(var_df.columns),
                "gene_names_sample": gene_names,
                "message": f"Found {n_genes} genes matching filter: {var_value_filter if var_value_filter else 'all genes'}"
            }

    def _get_anndata(
        self,
        organism: str,
        obs_value_filter: Optional[str],
        var_value_filter: Optional[str],
        obs_column_names: Optional[List[str]],
        var_column_names: Optional[List[str]],
        census_version: str,
        output_dir: str,
        output_id: str,
        max_cells: int
    ) -> Dict[str, Any]:
        """Retrieve expression data as AnnData."""
        # Default filter to avoid duplicates
        if obs_value_filter is None:
            obs_value_filter = "is_primary_data == True"

        # Add is_primary_data filter if not present
        if "is_primary_data" not in obs_value_filter:
            obs_value_filter = f"{obs_value_filter} and is_primary_data == True"

        # Default columns
        if obs_column_names is None:
            obs_column_names = ["cell_type", "tissue_general", "disease", "donor_id", "assay"]

        if var_column_names is None:
            var_column_names = ["feature_id", "feature_name", "feature_length"]

        with self._census.open_soma(census_version=census_version) as census:
            # First check cell count
            obs_df = self._census.get_obs(
                census,
                organism,
                value_filter=obs_value_filter,
                column_names=["soma_joinid"]
            )

            n_cells_available = len(obs_df)

            if n_cells_available > max_cells:
                logger.warning(f"Query would return {n_cells_available} cells, exceeding max_cells={max_cells}")
                logger.info("Consider using more specific filters or axis_query for large-scale processing")

                # Return metadata only without full data
                return {
                    "status": "too_large",
                    "n_cells_available": n_cells_available,
                    "max_cells": max_cells,
                    "suggestion": "Use more specific filters or axis_query for out-of-core processing",
                    "obs_value_filter": obs_value_filter,
                    "message": f"Query too large: {n_cells_available} cells > {max_cells} max. Please add more specific filters."
                }

            # Retrieve AnnData
            adata = self._census.get_anndata(
                census=census,
                organism=organism,
                obs_value_filter=obs_value_filter,
                var_value_filter=var_value_filter,
                obs_column_names=obs_column_names,
                var_column_names=var_column_names
            )

            n_cells = adata.n_obs
            n_genes = adata.n_vars

            # Save to h5ad
            output_file = os.path.join(output_dir, f"census_anndata_{output_id}.h5ad")
            adata.write_h5ad(output_file)

            # Get unique values summary
            unique_counts = {}
            for col in ["cell_type", "tissue_general", "disease"]:
                if col in adata.obs.columns:
                    unique_counts[col] = int(adata.obs[col].nunique())

            return {
                "status": "success",
                "output_file": output_file,
                "n_cells": n_cells,
                "n_genes": n_genes,
                "organism": organism,
                "obs_value_filter": obs_value_filter,
                "var_value_filter": var_value_filter,
                "unique_counts": unique_counts,
                "obs_columns": list(adata.obs.columns),
                "var_columns": list(adata.var.columns),
                "message": f"Retrieved {n_cells} cells x {n_genes} genes. Saved to {output_file}"
            }


if __name__ == "__main__":
    # Test the tool
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cellxgene_census_tool.py <operation>")
        print("Operations: get_summary, get_datasets, get_obs, get_var, get_anndata")
        sys.exit(1)

    operation = sys.argv[1]

    kwargs = {}
    for i in range(2, len(sys.argv), 2):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]
            value = sys.argv[i+1]
            kwargs[key] = value

    tool = CellxGeneCensusQuery()
    result, message = tool.run(operation=operation, **kwargs)
    print(f"Result: {result}")
    print(f"Message: {message}")