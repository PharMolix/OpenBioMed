"""
BoltzGen All-Atom Protein Design Tool.

A tool for all-atom protein/peptide design using BoltzGen diffusion model via external API.
Supports protein binder design, small molecule binding design, cyclic peptide design, and
nanobody/antibody CDR design.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import time
import logging
import requests
import json
import glob

from open_biomed.tools.base_tool import Tool, serial_exec

logger = logging.getLogger('OpenBioMed')

# API endpoint - configurable via environment variable
BOLTZGEN_API_BASE_URL = os.environ.get(
    "BOLTZGEN_API_BASE_URL",
    "http://172.16.20.44:10002"
)


class BoltzGenRequester(Tool):
    """
    BoltzGen all-atom protein design tool for de novo protein/peptide design.

    Supports:
    1. Protein binder design (protein-anything protocol)
    2. Small molecule binding design (protein-small_molecule protocol)
    3. Cyclic peptide design (peptide-anything protocol)
    4. Nanobody CDR design (nanobody-anything protocol)
    5. Antibody CDR design (antibody-anything protocol)

    Uses submit + poll pattern for asynchronous job execution (12-45 minutes).
    """

    def __init__(
        self,
        output_dir: str = "./tmp/boltzgen",
        timeout: int = 3600,
        poll_interval: int = 30
    ) -> None:
        self.output_dir = output_dir
        self.timeout = timeout
        self.poll_interval = poll_interval
        os.makedirs(self.output_dir, exist_ok=True)

    def print_usage(self) -> str:
        return "\n".join([
            'BoltzGen All-Atom Protein Design',
            'Inputs:',
            '  - yaml_file: Design YAML file path (required)',
            '  - protocol: Design protocol (default: protein-anything)',
            '  - num_designs: Number of intermediate designs (default: 10)',
            '  - budget: Final diversity-optimized set size (default: 2)',
            '  - cif_files: List of CIF/PDB target file paths (optional)',
            '  - output_name: Output file name prefix (optional)',
            'Outputs:',
            '  - design.cif: Final best all-atom design structure',
            '  - intermediate_designs/*.cif: Raw diffusion outputs',
            '  - intermediate_designs_inverse_folded/*.cif: Refolded complexes',
            '  - aggregate_metrics_analyze.csv: Quality metrics',
            '  - status.json: Pipeline status',
        ])

    def _health_check(self) -> bool:
        """Check BoltzGen API availability."""
        try:
            response = requests.get(
                f"{BOLTZGEN_API_BASE_URL}/health",
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"BoltzGen API healthy")
                return True
        except Exception as e:
            logger.warning(f"BoltzGen health check failed: {e}")
        return False

    def _submit_job(
        self,
        yaml_server_path: str,
        protocol: str,
        num_designs: int,
        budget: int,
        cif_server_paths: List[str] = None,
        extra_args: str = None
    ) -> str:
        """
        Submit design job to BoltzGen API.

        Args:
            yaml_server_path: Server path for design YAML file
            protocol: Design protocol
            num_designs: Number of intermediate designs
            budget: Final diversity-optimized set size
            cif_server_paths: Server paths for CIF/PDB target files
            extra_args: Extra CLI arguments

        Returns:
            job_id string
        """
        # Prepare multipart form data
        files = {}
        yaml_file_name = os.path.basename(yaml_server_path)
        files['design_yaml'] = (yaml_file_name, open(yaml_server_path, 'rb'))

        if cif_server_paths:
            for cif_path in cif_server_paths:
                cif_file_name = os.path.basename(cif_path)
                # Note: 'files' is used multiple times for multiple files
                files.setdefault('files', [])
                files['files'].append((cif_file_name, open(cif_path, 'rb')))

        # Prepare form data
        data = {
            'protocol': protocol,
            'num_designs': str(num_designs),
            'budget': str(budget)
        }
        if extra_args:
            data['extra_args'] = extra_args

        try:
            response = requests.post(
                f"{BOLTZGEN_API_BASE_URL}/jobs",
                files=files,
                data=data,
                timeout=60
            )

            # Close file handles
            for key, file_tuple in files.items():
                if isinstance(file_tuple, tuple):
                    file_tuple[1].close()
                elif isinstance(file_tuple, list):
                    for ft in file_tuple:
                        ft[1].close()

            if response.status_code == 503:
                raise Exception("BoltzGen queue is full (max 5 jobs). Please wait for existing jobs to finish.")
            elif response.status_code != 200:
                raise Exception(f"Submit job failed: {response.status_code} - {response.text}")

            result = response.json()
            job_id = result.get("job_id")
            logger.info(f"Submitted BoltzGen job: {job_id}")
            return job_id

        finally:
            # Ensure all file handles are closed
            for key, file_tuple in files.items():
                if isinstance(file_tuple, tuple):
                    if hasattr(file_tuple[1], 'close'):
                        file_tuple[1].close()
                elif isinstance(file_tuple, list):
                    for ft in file_tuple:
                        if hasattr(ft[1], 'close'):
                            ft[1].close()

    def _poll_status(self, job_id: str) -> Dict:
        """
        Poll job status until completed or failed.

        Args:
            job_id: Job ID string

        Returns:
            Job status dict
        """
        start_time = time.time()
        last_status = None

        while time.time() - start_time < self.timeout:
            response = requests.get(
                f"{BOLTZGEN_API_BASE_URL}/jobs/{job_id}",
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Poll status failed: {response.status_code}")
                time.sleep(self.poll_interval)
                continue

            result = response.json()
            status = result.get("status")

            # Log status change
            if status != last_status:
                logger.info(f"Job {job_id} status: {status}")
                last_status = status

            if status == "succeeded":
                logger.info(f"Job {job_id} completed successfully")
                return result
            elif status == "failed":
                error = result.get("error_message", "Unknown error")
                logger.error(f"Job {job_id} failed: {error}")
                raise Exception(f"Job failed: {error}")
            else:
                # pending or running
                time.sleep(self.poll_interval)

        raise Exception(f"Job {job_id} timeout after {self.timeout}s")

    def _download_results(
        self,
        job_id: str,
        output_name: str = None
    ) -> Tuple[List[str], Dict]:
        """
        Download result files from BoltzGen API.

        Args:
            job_id: Job ID string
            output_name: Output file name prefix

        Returns:
            Tuple of (output file paths, metadata dict)
        """
        if output_name is None:
            output_name = f"boltzgen_{job_id}"

        job_output_dir = os.path.join(self.output_dir, output_name)
        os.makedirs(job_output_dir, exist_ok=True)

        output_files = []
        metadata = {"job_id": job_id}

        # Download all results as zip
        try:
            response = requests.get(
                f"{BOLTZGEN_API_BASE_URL}/jobs/{job_id}/download",
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                zip_path = os.path.join(job_output_dir, "results.zip")
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Unzip the file
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(job_output_dir)

                # Remove zip file after extraction
                os.remove(zip_path)

                # Collect extracted files
                for root, dirs, files in os.walk(job_output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        output_files.append(file_path)

                logger.info(f"Downloaded and extracted results to {job_output_dir}")

                # Parse key result files
                design_cif = os.path.join(job_output_dir, "output", "design.cif")
                if os.path.exists(design_cif):
                    metadata["design_cif"] = design_cif
                    output_files.append(design_cif)

                metrics_csv = os.path.join(job_output_dir, "output", "intermediate_designs_inverse_folded", "aggregate_metrics_analyze.csv")
                if os.path.exists(metrics_csv):
                    metadata["metrics_csv"] = metrics_csv

            else:
                logger.warning(f"Download results failed: {response.status_code}")
                # Try to download individual files
                design_cif = self._download_single_file(job_id, "design.cif", job_output_dir)
                if design_cif:
                    output_files.append(design_cif)
                    metadata["design_cif"] = design_cif

        except Exception as e:
            logger.warning(f"Failed to download zip: {e}, trying individual files")
            design_cif = self._download_single_file(job_id, "design.cif", job_output_dir)
            if design_cif:
                output_files.append(design_cif)
                metadata["design_cif"] = design_cif

        return output_files, metadata

    def _download_single_file(
        self,
        job_id: str,
        file_name: str,
        output_dir: str
    ) -> Optional[str]:
        """
        Download a single file from BoltzGen API.

        Args:
            job_id: Job ID string
            file_name: File name to download
            output_dir: Output directory

        Returns:
            File path or None if failed
        """
        try:
            response = requests.get(
                f"{BOLTZGEN_API_BASE_URL}/jobs/{job_id}/files/{file_name}",
                timeout=60
            )

            if response.status_code == 200:
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'w') as f:
                    f.write(response.text)
                logger.info(f"Downloaded {file_name} to {file_path}")
                return file_path

        except Exception as e:
            logger.warning(f"Failed to download {file_name}: {e}")

        return None

    def _get_job_log(self, job_id: str) -> str:
        """
        Get job progress log.

        Args:
            job_id: Job ID string

        Returns:
            Log content string
        """
        try:
            response = requests.get(
                f"{BOLTZGEN_API_BASE_URL}/jobs/{job_id}/log",
                timeout=30
            )
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Failed to get job log: {e}")
        return ""

    @serial_exec
    def run(
        self,
        yaml_file: str,
        protocol: str = "protein-anything",
        num_designs: int = 10,
        budget: int = 2,
        cif_files: List[str] = None,
        extra_args: str = None,
        output_name: str = None,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Run BoltzGen all-atom protein design.

        Args:
            yaml_file: Design YAML file server path (required, from skill upload)
            protocol: Design protocol (default: protein-anything)
            num_designs: Number of intermediate designs (default: 10)
            budget: Final diversity-optimized set size (default: 2)
            cif_files: List of CIF/PDB target file server paths (optional)
            extra_args: Extra CLI arguments (optional)
            output_name: Output file name prefix (optional)

        Returns:
            Tuple of (file paths list, description messages list)
        """
        logger.info(f"Starting BoltzGen design with protocol: {protocol}")

        # Validate required inputs
        if not yaml_file:
            return [], ["Error: yaml_file is required for BoltzGen design"]

        # Health check (optional)
        healthy = self._health_check()
        if not healthy:
            logger.warning("BoltzGen API health check failed, proceeding anyway...")

        # Generate output name if not provided
        if output_name is None:
            output_name = f"boltzgen_{int(time.time() * 1000)}"

        try:
            # Submit job (yaml_file and cif_files are server paths from skill upload)
            job_id = self._submit_job(
                yaml_server_path=yaml_file,
                protocol=protocol,
                num_designs=num_designs,
                budget=budget,
                cif_server_paths=cif_files,
                extra_args=extra_args
            )

            # Poll for result
            job_status = self._poll_status(job_id)

            # Download results
            output_files, metadata = self._download_results(job_id, output_name)

            # Format description
            description = self._format_description(
                protocol=protocol,
                output_files=output_files,
                metadata=metadata,
                job_status=job_status
            )

            return output_files, [description]

        except Exception as e:
            logger.error(f"BoltzGen design failed: {e}")
            return [], [f"Error: {str(e)}"]

    def _format_description(
        self,
        protocol: str,
        output_files: List[str],
        metadata: Dict,
        job_status: Dict
    ) -> str:
        """Format output description."""
        protocol_display = protocol.replace("_", "-").capitalize()
        desc_parts = [f"BoltzGen {protocol_display} design completed."]

        # Job info
        job_id = metadata.get("job_id", "N/A")
        desc_parts.append(f"Job ID: {job_id}")

        # Output files
        design_cif = metadata.get("design_cif")
        if design_cif:
            desc_parts.append(f"Design structure: {design_cif}")

        metrics_csv = metadata.get("metrics_csv")
        if metrics_csv:
            desc_parts.append(f"Quality metrics: {metrics_csv}")

        desc_parts.append(f"Total output files: {len(output_files)}")

        # Job status info
        if job_status:
            return_code = job_status.get("return_code")
            if return_code == 0:
                desc_parts.append("Status: Success (return_code=0)")

        return "\n".join(desc_parts)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tool = BoltzGenRequester()

    # Test health check
    print("Testing BoltzGen health check...")
    healthy = tool._health_check()
    print(f"BoltzGen API healthy: {healthy}")

    # Test print usage
    print("\nUsage:")
    print(tool.print_usage())