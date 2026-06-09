"""
IgGM Antibody Design Tool.

A tool for de novo antibody design using IgGM external API.
Supports epitope-conditioned nanobody and heavy-light antibody design.
"""

from typing import Tuple, List, Dict, Any
import os
import time
import logging
import requests
import json
import base64

from open_biomed.tools.base_tool import Tool, serial_exec

logger = logging.getLogger('OpenBioMed')

# API endpoints - configurable via environment variables
IGGM_API_BASE_URL = os.environ.get(
    "IGGM_API_BASE_URL",
    "http://43.142.171.112:11280/IgGM"
)
PIPELINE_API_URL = os.environ.get(
    "PIPELINE_API_URL",
    "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
)


class IgGMRequester(Tool):
    """
    IgGM antibody design tool for epitope-conditioned de novo design.

    Supports:
    1. Nanobody design (single chain)
    2. Heavy-Light antibody design (two chains)
    """

    def __init__(
        self,
        output_dir: str = "./tmp/iggm",
        timeout: int = 300
    ) -> None:
        self.output_dir = output_dir
        self.timeout = timeout
        os.makedirs(self.output_dir, exist_ok=True)

    def print_usage(self) -> str:
        return "\n".join([
            'IgGM Antibody De Novo Design',
            'Inputs:',
            '  - design_type: "nanobody" or "heavy_light"',
            '  - antigen_pdb: Antigen PDB file path',
            '  - heavy_chain_mask: Heavy chain sequence with X for design regions',
            '  - light_chain_mask: Light chain sequence with X (for heavy_light)',
            '  - epitope: JSON list of epitope residue numbers, e.g., [109,110,111]',
            '  - num_samples: Number of design samples (default 1)',
            '  - steps: Sampling steps (default 10)',
            '  - antigen_chain_id: Antigen chain ID in PDB (default "A")',
            '  - output_name: Output file name (optional)',
            'Outputs:',
            '  - PDB files: Designed antibody structures',
            '  - FASTA files: Designed sequences',
            '  - JSON metadata: Job info and sequences'
        ])

    def _health_check(self) -> bool:
        """Check IgGM API availability."""
        try:
            response = requests.get(
                f"{IGGM_API_BASE_URL}/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"IgGM API healthy: {data}")
                return data.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"IgGM health check failed: {e}")
        return False

    def _design_antibody(
        self,
        antigen_pdb: str,
        heavy_chain_mask: str,
        light_chain_mask: str = None,
        epitope: List[int] = None,
        num_samples: int = 1,
        steps: int = 10,
        antigen_chain_id: str = "A",
        output_name: str = None
    ) -> Tuple[List[str], Dict]:
        """
        Design antibody or nanobody using IgGM API.

        Args:
            antigen_pdb: Path to antigen PDB file
            heavy_chain_mask: Heavy chain sequence with X marks for design regions
            light_chain_mask: Light chain sequence (None for nanobody)
            epitope: List of epitope residue numbers
            num_samples: Number of design samples
            steps: Sampling steps
            antigen_chain_id: Antigen chain ID in PDB
            output_name: Output file name prefix

        Returns:
            Tuple of (file paths list, metadata dict)
        """
        if output_name is None:
            output_name = f"iggm_design_{int(time.time() * 1000)}"

        if not os.path.exists(antigen_pdb):
            raise FileNotFoundError(f"Antigen PDB not found: {antigen_pdb}")

        if epitope is None or len(epitope) == 0:
            raise ValueError("Epitope residue list is required and must not be empty")

        logger.info(f"Calling IgGM API for antibody design...")
        logger.info(f"Antigen PDB: {antigen_pdb}")
        logger.info(f"Epitope residues: {epitope}")
        logger.info(f"Num samples: {num_samples}, Steps: {steps}")

        # Build multipart form data
        try:
            with open(antigen_pdb, 'rb') as f:
                files = {
                    "antigen_pdb": (os.path.basename(antigen_pdb), f, 'application/octet-stream')
                }
                data = {
                    "heavy_chain_mask": heavy_chain_mask,
                    "epitope": json.dumps(epitope),
                    "num_samples": str(num_samples),
                    "steps": str(steps),
                    "antigen_chain_id": antigen_chain_id
                }

                if light_chain_mask:
                    data["light_chain_mask"] = light_chain_mask

                response = requests.post(
                    f"{IGGM_API_BASE_URL}/design",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
        except requests.Timeout:
            logger.error("IgGM API request timed out")
            raise Exception("IgGM design request timed out")

        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"IgGM API error: {response.status_code} - {error_msg}")
            raise Exception(f"IgGM API returned {response.status_code}: {error_msg}")

        # Parse JSON response
        result = response.json()
        logger.info(f"Job ID: {result.get('job_id')}")
        logger.info(f"Antibody type: {result.get('antibody_type')}")

        # Decode and save files
        output_files = []

        # Save PDB files (base64 decoded)
        for i, pdb_file in enumerate(result.get("pdb_files", [])):
            filename = pdb_file.get("filename", f"output_{i}.pdb")
            pdb_path = os.path.join(self.output_dir, f"{output_name}_{i}.pdb")
            content_base64 = pdb_file.get("content_base64", "")
            if content_base64:
                content = base64.b64decode(content_base64)
                with open(pdb_path, 'wb') as f:
                    f.write(content)
                output_files.append(pdb_path)
                logger.info(f"PDB saved: {pdb_path}")

        # Save FASTA files (base64 decoded)
        for i, fasta_file in enumerate(result.get("fasta_files", [])):
            filename = fasta_file.get("filename", f"output_{i}.fasta")
            fasta_path = os.path.join(self.output_dir, f"{output_name}_{i}.fasta")
            content_base64 = fasta_file.get("content_base64", "")
            if content_base64:
                content = base64.b64decode(content_base64)
                with open(fasta_path, 'wb') as f:
                    f.write(content)
                output_files.append(fasta_path)
                logger.info(f"FASTA saved: {fasta_path}")

        # Save complete result JSON for reference
        result_json_path = os.path.join(self.output_dir, f"{output_name}_result.json")
        # Remove base64 content from saved JSON to reduce file size
        result_for_save = {
            "job_id": result.get("job_id"),
            "antibody_type": result.get("antibody_type"),
            "sequences": result.get("sequences", []),
            "pdb_files": [f.get("filename") for f in result.get("pdb_files", [])],
            "fasta_files": [f.get("filename") for f in result.get("fasta_files", [])],
            "saved_files": output_files
        }
        with open(result_json_path, 'w') as f:
            json.dump(result_for_save, f, indent=2)
        output_files.append(result_json_path)

        # Extract metadata
        metadata = {
            "job_id": result.get("job_id"),
            "antibody_type": result.get("antibody_type"),
            "sequences": result.get("sequences", []),
            "num_samples": len(result.get("pdb_files", []))
        }

        return output_files, metadata

    def _read_protein_file_local(self, pdb_file: str) -> Dict:
        """Read protein file locally."""
        try:
            from open_biomed.tools.file_reader_tools import ReadProteinFile
            reader = ReadProteinFile()
            results, messages = reader.run(protein=pdb_file, value="true")
            if results:
                return {
                    "sequence": results[0].sequence if hasattr(results[0], 'sequence') else "",
                    "pdb_content": messages[0] if messages else ""
                }
        except Exception as e:
            logger.warning(f"Could not read protein file locally: {e}")
        return {}

    def _read_protein_file_via_api(self, pdb_file: str) -> Dict:
        """Call run_pipeline to read protein content for remote agent."""
        try:
            response = requests.post(
                PIPELINE_API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "task": "read_protein_file",
                    "protein": pdb_file,
                    "value": "true"
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Protein file read successfully via API")
                return result
            else:
                logger.warning(f"read_protein_file API failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not read protein file via API: {e}")
        return {}

    @serial_exec
    def run(
        self,
        design_type: str = "nanobody",
        antigen_pdb: str = None,
        heavy_chain_mask: str = None,
        light_chain_mask: str = None,
        epitope: str = None,
        num_samples: int = 1,
        steps: int = 10,
        antigen_chain_id: str = "A",
        output_name: str = None,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Run IgGM antibody design.

        Args:
            design_type: "nanobody" or "heavy_light"
            antigen_pdb: Antigen PDB file path
            heavy_chain_mask: Heavy chain sequence with X marks
            light_chain_mask: Light chain sequence (for heavy_light)
            epitope: Epitope residues (JSON string, comma-separated string, or list)
            num_samples: Number of design samples
            steps: Sampling steps
            antigen_chain_id: Antigen chain ID
            output_name: Output file name prefix

        Returns:
            Tuple of (file paths list, description messages list)
        """
        logger.info(f"Starting IgGM design with design_type: {design_type}")

        # Parse epitope from various formats
        if epitope is None:
            return [], ["Error: epitope is required for antibody design"]

        epitope_list = None
        if isinstance(epitope, str):
            # Try JSON format first
            try:
                epitope_list = json.loads(epitope)
                if isinstance(epitope_list, int):
                    epitope_list = [epitope_list]
            except json.JSONDecodeError:
                # Try comma-separated format
                try:
                    epitope_list = [int(x.strip()) for x in epitope.split(",")]
                except ValueError:
                    return [], ["Error: epitope format invalid. Use JSON list like [109,110,111] or comma-separated like 109,110,111"]
        elif isinstance(epitope, list):
            epitope_list = epitope
        elif isinstance(epitope, int):
            epitope_list = [epitope]
        else:
            return [], ["Error: epitope must be a string (JSON/comma-separated), list, or int"]

        if not epitope_list:
            return [], ["Error: epitope list is empty"]

        # Health check (optional - continue even if fails)
        healthy = self._health_check()
        if not healthy:
            logger.warning("IgGM API health check failed, proceeding anyway...")

        try:
            # Validate required inputs
            if not antigen_pdb:
                return [], ["Error: antigen_pdb is required for antibody design"]

            if not heavy_chain_mask:
                return [], ["Error: heavy_chain_mask is required for antibody design"]

            if design_type == "heavy_light" and not light_chain_mask:
                return [], ["Error: light_chain_mask is required for heavy_light design type"]

            # Check for X marks in masks
            if "X" not in heavy_chain_mask:
                logger.warning("heavy_chain_mask contains no X marks - no design regions specified")

            if light_chain_mask and "X" not in light_chain_mask:
                logger.warning("light_chain_mask contains no X marks - no design regions specified")

            # Run design
            output_files, metadata = self._design_antibody(
                antigen_pdb=antigen_pdb,
                heavy_chain_mask=heavy_chain_mask,
                light_chain_mask=light_chain_mask if design_type == "heavy_light" else None,
                epitope=epitope_list,
                num_samples=num_samples,
                steps=steps,
                antigen_chain_id=antigen_chain_id,
                output_name=output_name
            )

            # Read first PDB for display
            pdb_files = [f for f in output_files if f.endswith(".pdb")]
            protein_info = {}
            if pdb_files:
                protein_info = self._read_protein_file_local(pdb_files[0])
                if not protein_info:
                    protein_info = self._read_protein_file_via_api(pdb_files[0])

            # Format description
            description = self._format_description(
                design_type, output_files, metadata, protein_info
            )

            return output_files, [description]

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return [], [f"Error: {str(e)}"]
        except Exception as e:
            logger.error(f"IgGM design failed: {e}")
            return [], [f"Error: {str(e)}"]

    def _format_description(
        self,
        design_type: str,
        output_files: List[str],
        metadata: Dict,
        protein_info: Dict
    ) -> str:
        """Format output description."""
        design_display = design_type.replace("_", "-").capitalize()
        desc_parts = [f"{design_display} antibody design completed."]

        # Job info
        job_id = metadata.get("job_id", "N/A")
        antibody_type = metadata.get("antibody_type", "unknown")
        desc_parts.append(f"Job ID: {job_id}")
        desc_parts.append(f"Antibody type: {antibody_type}")

        # Output files
        pdb_files = [f for f in output_files if f.endswith(".pdb")]
        fasta_files = [f for f in output_files if f.endswith(".fasta")]
        json_files = [f for f in output_files if f.endswith(".json")]

        desc_parts.append(f"Generated files: {len(output_files)}")
        if pdb_files:
            desc_parts.append(f"PDB files: {pdb_files[0]}")
        if fasta_files:
            desc_parts.append(f"FASTA files: {fasta_files[0]}")

        # Designed sequences
        sequences = metadata.get("sequences", [])
        if sequences:
            desc_parts.append(f"--- Design Sample 0 ---")
            seq = sequences[0]
            heavy = seq.get("heavy_chain", "")
            light = seq.get("light_chain")
            antigen_seq = seq.get("antigen", "")

            if heavy:
                desc_parts.append(f"Heavy chain: {heavy[:60]}...")
            if light:
                desc_parts.append(f"Light chain: {light[:60]}...")
            if antigen_seq:
                desc_parts.append(f"Antigen: {antigen_seq[:40]}...")

        # Protein info from reading PDB
        if protein_info:
            seq = protein_info.get("sequence", "")
            if seq:
                desc_parts.append(f"Design structure length: {len(seq)} residues")

        return "\n".join(desc_parts)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tool = IgGMRequester()

    # Test health check
    print("Testing IgGM health check...")
    healthy = tool._health_check()
    print(f"IgGM API healthy: {healthy}")

    # Test print usage
    print("\nUsage:")
    print(tool.print_usage())