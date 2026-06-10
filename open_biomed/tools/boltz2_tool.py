"""
Boltz-2 Structure Prediction Tool.

A tool for protein complex and protein-ligand structure prediction using Boltz-2 external API.
Supports affinity prediction (protein-ligand) and protein complex structure prediction.
"""

from typing import Tuple, List, Dict, Any
import os
import time
import logging
import requests
import json

from open_biomed.tools.base_tool import Tool, serial_exec

logger = logging.getLogger('OpenBioMed')

# API endpoint - configurable via environment variable
BOLTZ2_API_BASE_URL = os.environ.get(
    "BOLTZ2_API_BASE_URL",
    "http://172.16.20.44:17827/Boltz2"
)


class Boltz2Requester(Tool):
    """
    Boltz-2 structure prediction tool for protein complex and protein-ligand prediction.

    Supports:
    1. Protein-ligand affinity prediction (with structure generation)
    2. Protein complex structure prediction (two chains)

    Uses submit + poll pattern for asynchronous job execution.
    """

    def __init__(
        self,
        output_dir: str = "./tmp/boltz2",
        timeout: int = 600,
        poll_interval: int = 10
    ) -> None:
        self.output_dir = output_dir
        self.timeout = timeout
        self.poll_interval = poll_interval
        os.makedirs(self.output_dir, exist_ok=True)

    def print_usage(self) -> str:
        return "\n".join([
            'Boltz-2 Structure Prediction',
            'Inputs:',
            '  - prediction_type: "affinity" or "prot_complex"',
            '  - task_id: Project/batch ID for directory organization',
            '  - task_name: Task name for directory organization',
            '  - sequence: Protein sequence (for affinity)',
            '  - smiles: Ligand SMILES (for affinity)',
            '  - sequence_1: First protein sequence (for prot_complex)',
            '  - sequence_2: Second protein sequence (for prot_complex)',
            '  - output_name: Output file name prefix (optional)',
            'Outputs:',
            '  - PDB files: Predicted structures',
            '  - JSON files: Affinity scores (for affinity mode)',
        ])

    def _health_check(self) -> bool:
        """Check Boltz2 API availability."""
        try:
            response = requests.get(
                f"{BOLTZ2_API_BASE_URL}/list_jobs",
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"Boltz2 API healthy")
                return True
        except Exception as e:
            logger.warning(f"Boltz2 health check failed: {e}")
        return False

    def _submit_affinity(
        self,
        task_id: str,
        task_name: str,
        sequence: str,
        smiles: str
    ) -> str:
        """
        Submit protein-ligand affinity prediction job.

        Returns:
            job_id string
        """
        response = requests.post(
            f"{BOLTZ2_API_BASE_URL}/submit_affinity",
            headers={"Content-Type": "application/json"},
            json={
                "task_id": task_id,
                "task_name": task_name,
                "sequence": sequence,
                "smiles": smiles
            },
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Submit affinity failed: {response.status_code} - {response.text}")

        result = response.json()
        job_id = result.get("job_id")
        logger.info(f"Submitted affinity job: {job_id}")
        return job_id

    def _submit_prot_complex(
        self,
        task_id: str,
        task_name: str,
        sequence_1: str,
        sequence_2: str
    ) -> str:
        """
        Submit protein complex structure prediction job.

        Returns:
            job_id string
        """
        response = requests.post(
            f"{BOLTZ2_API_BASE_URL}/submit_prot_complex",
            headers={"Content-Type": "application/json"},
            json={
                "task_id": task_id,
                "task_name": task_name,
                "sequence_1": sequence_1,
                "sequence_2": sequence_2
            },
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Submit prot_complex failed: {response.status_code} - {response.text}")

        result = response.json()
        job_id = result.get("job_id")
        logger.info(f"Submitted prot_complex job: {job_id}")
        return job_id

    def _poll_result(self, job_id: str) -> Dict:
        """
        Poll job result until completed or failed.

        Returns:
            Result dict with status, result, error fields
        """
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            response = requests.get(
                f"{BOLTZ2_API_BASE_URL}/fetch_result",
                params={"job_id": job_id},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Poll result failed: {response.status_code}")
                time.sleep(self.poll_interval)
                continue

            result = response.json()
            status = result.get("status")

            if status == "completed":
                logger.info(f"Job {job_id} completed")
                return result
            elif status == "failed":
                error = result.get("error", "Unknown error")
                logger.error(f"Job {job_id} failed: {error}")
                raise Exception(f"Job failed: {error}")
            else:
                # pending or running
                logger.info(f"Job {job_id} status: {status}, polling...")
                time.sleep(self.poll_interval)

        raise Exception(f"Job {job_id} timeout after {self.timeout}s")

    def _save_result(
        self,
        job_id: str,
        result: Dict,
        output_name: str = None
    ) -> Tuple[List[str], Dict]:
        """
        Save prediction result to files.

        Returns:
            Tuple of (file paths list, metadata dict)
        """
        if output_name is None:
            output_name = f"boltz2_{int(time.time() * 1000)}"

        output_files = []
        metadata = {"job_id": job_id}

        # Save structure PDB
        structure = result.get("result", {}).get("structure")
        if structure:
            pdb_path = os.path.join(self.output_dir, f"{output_name}.pdb")
            with open(pdb_path, 'w') as f:
                f.write(structure)
            output_files.append(pdb_path)
            logger.info(f"Structure saved: {pdb_path}")
            metadata["structure_file"] = pdb_path

        # Save affinity data (for affinity mode)
        affinity_result = result.get("result", {})
        if "affinity" in affinity_result:
            affinity_data = {
                "affinity": affinity_result.get("affinity"),
                "ic50": affinity_result.get("ic50")
            }
            metadata["affinity"] = affinity_data

            # Save affinity JSON
            affinity_path = os.path.join(self.output_dir, f"{output_name}_affinity.json")
            with open(affinity_path, 'w') as f:
                json.dump(affinity_data, f, indent=2)
            output_files.append(affinity_path)
            logger.info(f"Affinity saved: {affinity_path}")

        # Save complete result JSON
        result_path = os.path.join(self.output_dir, f"{output_name}_result.json")
        result_for_save = {
            "job_id": job_id,
            "status": result.get("status"),
            "result": result.get("result"),
            "created_at": result.get("created_at"),
            "finished_at": result.get("finished_at"),
            "saved_files": output_files
        }
        with open(result_path, 'w') as f:
            json.dump(result_for_save, f, indent=2)
        output_files.append(result_path)

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

    @serial_exec
    def run(
        self,
        prediction_type: str = "affinity",
        task_id: str = None,
        task_name: str = None,
        sequence: str = None,
        smiles: str = None,
        sequence_1: str = None,
        sequence_2: str = None,
        output_name: str = None,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Run Boltz-2 structure prediction.

        Args:
            prediction_type: "affinity" or "prot_complex"
            task_id: Project/batch ID
            task_name: Task name
            sequence: Protein sequence (for affinity)
            smiles: Ligand SMILES (for affinity)
            sequence_1: First protein sequence (for prot_complex)
            sequence_2: Second protein sequence (for prot_complex)
            output_name: Output file name prefix

        Returns:
            Tuple of (file paths list, description messages list)
        """
        logger.info(f"Starting Boltz2 prediction with type: {prediction_type}")

        # Generate default task_id/task_name if not provided
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"
        if task_name is None:
            task_name = f"{prediction_type}_{int(time.time() * 1000)}"

        # Health check (optional)
        healthy = self._health_check()
        if not healthy:
            logger.warning("Boltz2 API health check failed, proceeding anyway...")

        try:
            # Validate required inputs
            if prediction_type == "affinity":
                if not sequence:
                    return [], ["Error: sequence is required for affinity prediction"]
                if not smiles:
                    return [], ["Error: smiles is required for affinity prediction"]

                job_id = self._submit_affinity(
                    task_id=task_id,
                    task_name=task_name,
                    sequence=sequence,
                    smiles=smiles
                )

            elif prediction_type == "prot_complex":
                if not sequence_1:
                    return [], ["Error: sequence_1 is required for prot_complex prediction"]
                if not sequence_2:
                    return [], ["Error: sequence_2 is required for prot_complex prediction"]

                job_id = self._submit_prot_complex(
                    task_id=task_id,
                    task_name=task_name,
                    sequence_1=sequence_1,
                    sequence_2=sequence_2
                )

            else:
                return [], [f"Error: Unknown prediction_type: {prediction_type}. Use 'affinity' or 'prot_complex'"]

            # Poll for result
            result = self._poll_result(job_id)

            # Save result files
            output_files, metadata = self._save_result(job_id, result, output_name)

            # Read PDB for display
            pdb_files = [f for f in output_files if f.endswith(".pdb")]
            protein_info = {}
            if pdb_files:
                protein_info = self._read_protein_file_local(pdb_files[0])

            # Format description
            description = self._format_description(prediction_type, output_files, metadata, protein_info)

            return output_files, [description]

        except Exception as e:
            logger.error(f"Boltz2 prediction failed: {e}")
            return [], [f"Error: {str(e)}"]

    def _format_description(
        self,
        prediction_type: str,
        output_files: List[str],
        metadata: Dict,
        protein_info: Dict
    ) -> str:
        """Format output description."""
        pred_display = prediction_type.replace("_", "-").capitalize()
        desc_parts = [f"Boltz-2 {pred_display} prediction completed."]

        # Job info
        job_id = metadata.get("job_id", "N/A")
        desc_parts.append(f"Job ID: {job_id}")

        # Output files
        pdb_files = [f for f in output_files if f.endswith(".pdb")]
        json_files = [f for f in output_files if f.endswith(".json")]

        desc_parts.append(f"Generated files: {len(output_files)}")
        if pdb_files:
            desc_parts.append(f"Structure: {pdb_files[0]}")
        if json_files:
            desc_parts.append(f"Metadata: {json_files[0]}")

        # Affinity data (for affinity mode)
        if prediction_type == "affinity":
            affinity_data = metadata.get("affinity", {})
            if affinity_data:
                affinity_value = affinity_data.get("affinity")
                ic50_value = affinity_data.get("ic50")
                if affinity_value is not None:
                    desc_parts.append(f"Affinity prediction: {affinity_value:.4f}")
                if ic50_value is not None:
                    desc_parts.append(f"IC50: {ic50_value:.2f} nM")

        # Protein info
        if protein_info:
            seq = protein_info.get("sequence", "")
            if seq:
                desc_parts.append(f"Structure length: {len(seq)} residues")

        return "\n".join(desc_parts)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tool = Boltz2Requester()

    # Test health check
    print("Testing Boltz2 health check...")
    healthy = tool._health_check()
    print(f"Boltz2 API healthy: {healthy}")

    # Test print usage
    print("\nUsage:")
    print(tool.print_usage())