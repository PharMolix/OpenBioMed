"""
Spatial transcriptomics data loading tool.
Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, CosMx, Stereo-seq platforms.
"""

import os
import uuid
import logging
from typing import Tuple, Dict, Any, Optional

from open_biomed.tools.base_tool import Tool

logger = logging.getLogger('OpenBioMed')


class SpatialTranscriptomicsLoader(Tool):
    """
    Load spatial transcriptomics data from various platforms.

    Supported platforms:
    - visium: 10x Genomics Visium (Space Ranger output)
    - xenium: 10x Genomics Xenium
    - merscope: Vizgen MERFISH/MERSCOPE
    - slideseq: Slide-seq / Slide-seqV2
    - cosmx: Nanostring CosMx Spatial Molecular Imager
    - stereoseq: BGI Stereo-seq

    Returns AnnData or SpatialData object with expression matrix,
    spatial coordinates, and tissue images.
    """

    def __init__(self) -> None:
        super().__init__()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import squidpy as sq
            import anndata as ad
            self._squidpy_available = True
        except ImportError as e:
            logger.warning(f"Missing dependency: {e}. Some features may not work.")
            self._squidpy_available = False

        # Note: spatialdata_io requires Python 3.10+ due to readfcs type annotation syntax
        # It will be imported lazily when needed for cosmx/stereoseq platforms
        # Do not import here to avoid Python 3.9 compatibility issues
        self._spatialdata_available = False
        self._spatialdata_io_available = False

    def print_usage(self) -> str:
        return """
Usage: Load spatial transcriptomics data from various platforms
Inputs: {
    "data_dir": str (path to platform-specific output directory),
    "platform": str (visium, xenium, merscope, slideseq, cosmx, stereoseq),
    "output_format": str (anndata or spatialdata, default: anndata),
    "library_id": str (optional, for Visium data)
}
Outputs: {
    "data_file": str (path to saved .h5ad or .zarr file),
    "platform": str,
    "n_obs": int (number of spots/cells),
    "n_vars": int (number of genes),
    "has_spatial_coords": bool,
    "has_images": bool,
    "description": str
}
"""

    def run(
        self,
        data_dir: str,
        platform: str,
        output_format: str = "anndata",
        library_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Load spatial transcriptomics data.

        Args:
            data_dir: Path to platform-specific output directory
            platform: Platform type (visium, xenium, merscope, slideseq, cosmx, stereoseq)
            output_format: Output format (anndata or spatialdata)
            library_id: Optional library ID for Visium data

        Returns:
            Tuple of (result_dict, message)
        """
        # Validate inputs
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        platform = platform.lower()
        supported_platforms = ["visium", "xenium", "merscope", "slideseq", "cosmx", "stereoseq"]
        if platform not in supported_platforms:
            raise ValueError(f"Unsupported platform: {platform}. Supported: {supported_platforms}")

        # Load data based on platform
        logger.info(f"Loading {platform} data from {data_dir}")

        if platform == "visium":
            adata = self._load_visium(data_dir, library_id)
        elif platform == "xenium":
            adata = self._load_xenium(data_dir)
        elif platform == "merscope":
            adata = self._load_merscope(data_dir)
        elif platform == "slideseq":
            adata = self._load_slideseq(data_dir)
        elif platform == "cosmx":
            adata = self._load_cosmx(data_dir)
        elif platform == "stereoseq":
            adata = self._load_stereoseq(data_dir)
        else:
            raise ValueError(f"Platform {platform} not implemented")

        # Generate output file
        output_id = str(uuid.uuid4())[:8]

        if output_format == "spatialdata":
            # Try to use SpatialData if available
            # Note: spatialdata_io may fail on Python 3.9 due to readfcs
            try:
                import spatialdata as sd

                # Convert AnnData to SpatialData if needed
                sdata = self._convert_to_spatialdata(adata, platform)
                output_file = f"./tmp/spatial_data_{output_id}.zarr"
                sdata.write(output_file)
                logger.info(f"Saved SpatialData to {output_file}")
            except ImportError:
                logger.warning("SpatialData not available, falling back to AnnData format")
                output_format = "anndata"
            except Exception as e:
                logger.warning(f"SpatialData conversion failed: {e}, falling back to AnnData format")
                output_format = "anndata"

        if output_format == "anndata":
            output_file = f"./tmp/spatial_data_{output_id}.h5ad"
            adata.write_h5ad(output_file)
            logger.info(f"Saved AnnData to {output_file}")

        # Build result
        has_spatial_coords = "spatial" in adata.obsm
        has_images = "spatial" in adata.uns if hasattr(adata, 'uns') and adata.uns is not None else False

        result = {
            "data_file": output_file,
            "platform": platform,
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "has_spatial_coords": has_spatial_coords,
            "has_images": has_images,
            "output_format": output_format
        }

        # Add spatial info if available
        if has_spatial_coords:
            result["spatial_coords_shape"] = list(adata.obsm["spatial"].shape)

        message = f"Loaded {platform} data: {adata.n_obs} spots/cells, {adata.n_vars} genes. Saved to {output_file}"

        return result, message

    def _load_visium(self, data_dir: str, library_id: Optional[str] = None) -> Any:
        """Load 10x Visium data from Space Ranger output."""
        import squidpy as sq

        # Check for required files
        filtered_matrix = os.path.join(data_dir, "filtered_feature_bc_matrix.h5")
        if not os.path.exists(filtered_matrix):
            # Try alternative locations
            filtered_matrix = os.path.join(data_dir, "outs", "filtered_feature_bc_matrix.h5")

        tissue_positions = os.path.join(data_dir, "tissue_positions_list.csv")
        if not os.path.exists(tissue_positions):
            tissue_positions = os.path.join(data_dir, "spatial", "tissue_positions_list.csv")

        if not os.path.exists(filtered_matrix):
            raise FileNotFoundError(
                f"Visium data not found. Expected filtered_feature_bc_matrix.h5 in {data_dir}"
            )

        # Load using squidpy
        try:
            adata = sq.read.visium(data_dir, library_id=library_id)
        except Exception as e:
            # Fallback to scanpy if squidpy fails
            import scanpy as sc
            adata = sc.read_visium(data_dir)

        return adata

    def _load_xenium(self, data_dir: str) -> Any:
        """Load 10x Xenium single-cell resolution data."""
        import squidpy as sq

        # Check for required files
        cell_summary = os.path.join(data_dir, "cells_summary.parquet")
        if not os.path.exists(cell_summary):
            cell_summary = os.path.join(data_dir, "outs", "cells_summary.parquet")

        if not os.path.exists(os.path.join(data_dir, "cells_summary.parquet")) and \
           not os.path.exists(os.path.join(data_dir, "outs", "cells_summary.parquet")):
            raise FileNotFoundError(
                f"Xenium data not found. Expected cells_summary.parquet in {data_dir}"
            )

        # Load using squidpy
        try:
            adata = sq.read.xenium(data_dir)
        except Exception as e:
            # Try spatialdata_io
            try:
                import spatialdata_io as sdio
                sdata = sdio.xenium(data_dir)
                adata = sdata.tables['table']
            except Exception as e2:
                raise RuntimeError(f"Failed to load Xenium data: {e}, {e2}")

        return adata

    def _load_merscope(self, data_dir: str) -> Any:
        """Load Vizgen MERFISH/MERSCOPE data."""
        try:
            import squidpy as sq

            # Check for required files
            counts_file = os.path.join(data_dir, "cell_by_gene.csv")
            meta_file = os.path.join(data_dir, "cell_metadata.csv")

            if not os.path.exists(counts_file):
                counts_file = os.path.join(data_dir, "outs", "cell_by_gene.csv")
                meta_file = os.path.join(data_dir, "outs", "cell_metadata.csv")

            if not os.path.exists(counts_file):
                raise FileNotFoundError(
                    f"MERSCOPE data not found. Expected cell_by_gene.csv in {data_dir}"
                )

            adata = sq.read.vizgen(
                data_dir,
                counts_file=counts_file,
                meta_file=meta_file
            )
        except Exception as e:
            # Try spatialdata_io
            try:
                import spatialdata_io as sdio
                sdata = sdio.merscope(data_dir)
                adata = sdata.tables['table']
            except Exception as e2:
                raise RuntimeError(f"Failed to load MERSCOPE data: {e}, {e2}")

        return adata

    def _load_slideseq(self, data_dir: str) -> Any:
        """Load Slide-seq / Slide-seqV2 data."""
        import squidpy as sq

        # Look for bead and coordinate files
        beads_file = None
        coords_file = None

        for f in os.listdir(data_dir):
            if 'bead' in f.lower() and f.endswith('.csv'):
                beads_file = os.path.join(data_dir, f)
            if 'coord' in f.lower() and f.endswith('.csv'):
                coords_file = os.path.join(data_dir, f)

        if beads_file is None:
            raise FileNotFoundError(
                f"Slide-seq bead file not found in {data_dir}"
            )

        try:
            adata = sq.read.slideseq(beads_file, coordinates_file=coords_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load Slide-seq data: {e}")

        return adata

    def _load_cosmx(self, data_dir: str) -> Any:
        """Load Nanostring CosMx Spatial Molecular Imager data."""
        try:
            import spatialdata_io as sdio

            sdata = sdio.cosmx(data_dir)
            adata = sdata.tables['table']
        except ImportError:
            raise ImportError(
                "spatialdata_io is required for CosMx data loading. "
                "Install with: pip install spatialdata-io (requires Python 3.10+)"
            )
        except TypeError as e:
            # readfcs requires Python 3.10+ for type annotation syntax
            raise RuntimeError(
                "CosMx data loading requires Python 3.10+ due to spatialdata_io/readfcs dependency. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CosMx data: {e}")

        return adata

    def _load_stereoseq(self, data_dir: str) -> Any:
        """Load BGI Stereo-seq data."""
        try:
            import spatialdata_io as sdio

            sdata = sdio.stereoseq(data_dir)
            adata = sdata.tables['table']
        except ImportError:
            raise ImportError(
                "spatialdata_io is required for Stereo-seq data loading. "
                "Install with: pip install spatialdata-io (requires Python 3.10+)"
            )
        except TypeError as e:
            # readfcs requires Python 3.10+ for type annotation syntax
            raise RuntimeError(
                "Stereo-seq data loading requires Python 3.10+ due to spatialdata_io/readfcs dependency. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Stereo-seq data: {e}")

        return adata

    def _convert_to_spatialdata(self, adata: Any, platform: str) -> Any:
        """Convert AnnData to SpatialData object."""
        import spatialdata as sd
        import numpy as np

        # Create SpatialData from AnnData
        # This is a simplified conversion - actual conversion may need more handling
        sdata = sd.SpatialData(tables={'table': adata})

        # Add shapes if spatial coordinates exist
        if "spatial" in adata.obsm:
            coords = adata.obsm["spatial"]
            # Create spot/cell shapes based on platform
            if platform in ["visium"]:
                # Visium uses circular spots
                from spatialdata.models import ShapesModel
                import pandas as pd
                spots_df = pd.DataFrame(
                    coords,
                    index=adata.obs_names,
                    columns=['x', 'y']
                )
                # Note: This is simplified; actual implementation needs proper radius handling
                sdata.shapes['spots'] = ShapesModel.parse(spots_df)

        return sdata


if __name__ == "__main__":
    # Test the tool
    import sys

    if len(sys.argv) < 3:
        print("Usage: python spatial_transcriptomics_tool.py <data_dir> <platform>")
        print("Platforms: visium, xenium, merscope, slideseq, cosmx, stereoseq")
        sys.exit(1)

    data_dir = sys.argv[1]
    platform = sys.argv[2]

    tool = SpatialTranscriptomicsLoader()
    result, message = tool.run(data_dir, platform)
    print(f"Result: {result}")
    print(f"Message: {message}")