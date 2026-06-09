"""
tFold Antibody Structure Prediction Tool.

A tool for predicting antibody structure, antigen-antibody complex,
and epitope residues using external tFold API.
"""

from typing import Tuple, List, Dict, Any
import os
import time
import logging
import requests
import json

from open_biomed.tools.base_tool import Tool, serial_exec

logger = logging.getLogger('OpenBioMed')

# API endpoints - configurable via environment variables
TFOLD_API_BASE_URL = os.environ.get(
    "TFOLD_API_BASE_URL",
    "http://172.16.20.26:11280/tFold"
)
PIPELINE_API_URL = os.environ.get(
    "PIPELINE_API_URL",
    "http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520/run_pipeline/"
)


class TFoldRequester(Tool):
    """
    tFold antibody structure prediction tool.

    Supports:
    1. Antibody structure prediction (heavy + light chain)
    2. Nanobody structure prediction (single chain)
    3. Antigen-antibody complex prediction
    4. Epitope determination from complex PDB
    """

    def __init__(
        self,
        output_dir: str = "./tmp/tfold",
        timeout: int = 300
    ) -> None:
        self.output_dir = output_dir
        self.timeout = timeout
        os.makedirs(self.output_dir, exist_ok=True)

    def print_usage(self) -> str:
        return "\n".join([
            'tFold Antibody Structure Prediction',
            'Inputs:',
            '  - prediction_type: "antibody", "nanobody", "complex", or "epitope"',
            '  - heavy_chain: Heavy chain FASTA sequence',
            '  - light_chain: Light chain FASTA sequence (for antibody/complex)',
            '  - antigen: Antigen FASTA sequence (for complex)',
            '  - pdb_file: PDB file path (for epitope)',
            '  - antigen_id: Chain ID for antigen (default: "A")',
            '  - msa_content: MSA content in a3m format (optional)',
            '  - distance_threshold: Distance threshold for epitope (default: 5.0)',
            '  - output_name: Output file name (optional)',
            'Outputs:',
            '  - PDB file path and confidence scores'
        ])

    def _health_check(self) -> bool:
        """Check tFold API availability."""
        try:
            response = requests.get(
                f"{TFOLD_API_BASE_URL}/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"tFold API healthy: {data}")
                return data.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"tFold health check failed: {e}")
        return False

    def _predict_antibody(
        self,
        heavy_chain: str,
        light_chain: str = None,
        output_name: str = None
    ) -> Tuple[str, Dict]:
        """
        Predict antibody or nanobody structure.

        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence (None for nanobody)
            output_name: Output file name

        Returns:
            Tuple of (pdb_file_path, metadata)
        """
        if output_name is None:
            output_name = f"tfold_{int(time.time() * 1000)}"

        # Build chains payload
        chains = [{"id": "H", "sequence": heavy_chain}]
        if light_chain:
            chains.append({"id": "L", "sequence": light_chain})

        payload = {
            "chains": chains,
            "output_name": output_name
        }

        logger.info(f"Calling tFold API for antibody prediction...")
        logger.info(f"Payload: chains={len(chains)}, output_name={output_name}")

        # Call tFold API
        try:
            response = requests.post(
                f"{TFOLD_API_BASE_URL}/predict/ab",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                pdb_file = os.path.join(self.output_dir, f"{output_name}.pdb")
                with open(pdb_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"PDB saved to {pdb_file}")

                # Extract confidence scores from PDB
                metadata = self._extract_confidence_scores(pdb_file)
                return pdb_file, metadata
            else:
                error_msg = response.text
                logger.error(f"tFold API error: {response.status_code} - {error_msg}")
                raise Exception(f"tFold API returned {response.status_code}: {error_msg}")

        except requests.Timeout:
            logger.error("tFold API timeout")
            raise Exception("tFold API request timed out")

    def _predict_complex(
        self,
        heavy_chain: str,
        light_chain: str,
        antigen: str,
        antigen_id: str = "A",
        msa_content: str = None,
        output_name: str = None
    ) -> Tuple[str, Dict]:
        """
        Predict antigen-antibody complex structure.

        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            antigen: Antigen sequence
            antigen_id: Chain ID for antigen
            msa_content: MSA content (optional)
            output_name: Output file name

        Returns:
            Tuple of (pdb_file_path, metadata)
        """
        if output_name is None:
            output_name = f"tfold_complex_{int(time.time() * 1000)}"

        payload = {
            "antibody_chains": [
                {"id": "H", "sequence": heavy_chain},
                {"id": "L", "sequence": light_chain}
            ],
            "antigen_sequence": antigen,
            "antigen_id": antigen_id,
            "output_name": output_name
        }

        if msa_content:
            payload["msa_content"] = msa_content

        logger.info(f"Calling tFold API for complex prediction...")
        logger.info(f"Antigen chain ID: {antigen_id}, output_name: {output_name}")

        try:
            response = requests.post(
                f"{TFOLD_API_BASE_URL}/predict/ag",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                pdb_file = os.path.join(self.output_dir, f"{output_name}.pdb")
                with open(pdb_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Complex PDB saved to {pdb_file}")

                metadata = self._extract_confidence_scores(pdb_file)
                return pdb_file, metadata
            else:
                error_msg = response.text
                logger.error(f"tFold complex API error: {response.status_code} - {error_msg}")
                raise Exception(f"tFold API returned {response.status_code}: {error_msg}")

        except requests.Timeout:
            logger.error("tFold complex prediction timed out")
            raise Exception("tFold complex prediction timed out")

    def _predict_epitope(
        self,
        pdb_file: str,
        antigen_id: str = "A",
        distance_threshold: float = 5.0
    ) -> Dict:
        """
        Determine epitope residues from complex PDB.

        Args:
            pdb_file: Path to complex PDB file
            antigen_id: Chain ID for antigen
            distance_threshold: Distance threshold in Angstroms

        Returns:
            Dict with epitope residue information
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        logger.info(f"Calling tFold API for epitope prediction...")
        logger.info(f"PDB file: {pdb_file}, antigen_id: {antigen_id}, threshold: {distance_threshold}")

        try:
            with open(pdb_file, 'rb') as f:
                files = {"pdb_file": (os.path.basename(pdb_file), f, 'application/octet-stream')}
                data = {
                    "antigen_id": antigen_id,
                    "distance_threshold": str(distance_threshold)
                }
                response = requests.post(
                    f"{TFOLD_API_BASE_URL}/predict/epitope",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Epitope prediction completed: {result.get('epitope_count', 0)} residues")
                return result
            else:
                error_msg = response.text
                logger.error(f"Epitope API error: {response.status_code} - {error_msg}")
                raise Exception(f"Epitope API returned {response.status_code}: {error_msg}")

        except requests.Timeout:
            logger.error("Epitope prediction timed out")
            raise Exception("Epitope prediction timed out")

    def _extract_confidence_scores(self, pdb_file: str) -> Dict:
        """Extract confidence scores from PDB REMARK lines."""
        metadata = {}
        try:
            with open(pdb_file, 'r') as f:
                content = f.read()

            # Parse REMARK 250 lines for confidence scores
            for line in content.split('\n'):
                if 'REMARK 250' in line:
                    if 'lDDT-Ca score' in line:
                        try:
                            score_str = line.split(':')[1].strip()
                            metadata['lddt_ca'] = float(score_str)
                        except (ValueError, IndexError):
                            pass
                    elif 'pTM score' in line:
                        try:
                            score_str = line.split(':')[1].strip()
                            metadata['ptm'] = float(score_str)
                        except (ValueError, IndexError):
                            pass
                    elif 'ipTM score' in line:
                        try:
                            score_str = line.split(':')[1].strip()
                            metadata['iptm'] = float(score_str)
                        except (ValueError, IndexError):
                            pass

            if metadata:
                logger.info(f"Confidence scores: {metadata}")
        except Exception as e:
            logger.warning(f"Could not extract confidence scores: {e}")

        return metadata

    def _read_protein_file_via_api(self, pdb_file: str) -> Dict:
        """Call run_pipeline to read and display protein content for remote agent."""
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
                return {}
        except Exception as e:
            logger.warning(f"Could not read protein file via API: {e}")
            return {}

    def _read_protein_file_local(self, pdb_file: str) -> Dict:
        """Read protein file locally when run_pipeline is available."""
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
        prediction_type: str = "antibody",
        heavy_chain: str = None,
        light_chain: str = None,
        antigen: str = None,
        pdb_file: str = None,
        antigen_id: str = "A",
        msa_content: str = None,
        distance_threshold: float = 5.0,
        output_name: str = None,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Run tFold prediction.

        Args:
            prediction_type: Type of prediction (antibody, nanobody, complex, epitope)
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            antigen: Antigen sequence
            pdb_file: PDB file path (for epitope)
            antigen_id: Chain ID for antigen
            msa_content: MSA content
            distance_threshold: Distance threshold for epitope
            output_name: Output file name

        Returns:
            Tuple of (result paths list, description messages list)
        """
        logger.info(f"Starting tFold prediction with prediction_type: {prediction_type}")

        # Health check (optional - continue even if fails)
        healthy = self._health_check()
        if not healthy:
            logger.warning("tFold API health check failed, proceeding anyway...")

        try:
            if prediction_type in ["antibody", "nanobody"]:
                if not heavy_chain:
                    return [], ["Error: heavy_chain is required for antibody/nanobody prediction"]

                if prediction_type == "antibody" and not light_chain:
                    return [], ["Error: light_chain is required for antibody prediction"]

                pdb_path, metadata = self._predict_antibody(
                    heavy_chain=heavy_chain,
                    light_chain=light_chain if prediction_type == "antibody" else None,
                    output_name=output_name
                )

                # Read protein for display
                protein_info = self._read_protein_file_local(pdb_path)
                if not protein_info:
                    protein_info = self._read_protein_file_via_api(pdb_path)

                description = self._format_antibody_description(
                    prediction_type, pdb_path, metadata, protein_info
                )
                return [pdb_path], [description]

            elif prediction_type == "complex":
                if not heavy_chain or not light_chain or not antigen:
                    return [], ["Error: heavy_chain, light_chain, and antigen are required for complex prediction"]

                pdb_path, metadata = self._predict_complex(
                    heavy_chain=heavy_chain,
                    light_chain=light_chain,
                    antigen=antigen,
                    antigen_id=antigen_id,
                    msa_content=msa_content,
                    output_name=output_name
                )

                protein_info = self._read_protein_file_local(pdb_path)
                if not protein_info:
                    protein_info = self._read_protein_file_via_api(pdb_path)

                description = self._format_complex_description(
                    pdb_path, metadata, protein_info, antigen_id
                )
                return [pdb_path], [description]

            elif prediction_type == "epitope":
                if not pdb_file:
                    return [], ["Error: pdb_file is required for epitope prediction"]

                result = self._predict_epitope(
                    pdb_file=pdb_file,
                    antigen_id=antigen_id,
                    distance_threshold=distance_threshold
                )

                # Save result to JSON file
                timestamp = int(time.time() * 1000)
                result_file = os.path.join(self.output_dir, f"epitope_result_{timestamp}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

                description = self._format_epitope_description(
                    result, antigen_id, distance_threshold
                )
                return [result_file], [description]

            else:
                return [], [f"Error: Unknown prediction_type '{prediction_type}'. Supported: antibody, nanobody, complex, epitope"]

        except Exception as e:
            logger.error(f"tFold prediction failed: {e}")
            return [], [f"Error: {str(e)}"]

    def _format_antibody_description(
        self,
        task: str,
        pdb_path: str,
        metadata: Dict,
        protein_info: Dict
    ) -> str:
        """Format output description for antibody/nanobody prediction."""
        task_display = task.capitalize()
        desc_parts = [f"{task_display} structure prediction completed."]
        desc_parts.append(f"PDB file: {pdb_path}")

        if metadata:
            if 'lddt_ca' in metadata:
                desc_parts.append(f"lDDT-Ca score: {metadata['lddt_ca']:.4f}")
            if 'ptm' in metadata:
                desc_parts.append(f"pTM score: {metadata['ptm']:.4f}")
            if 'iptm' in metadata:
                desc_parts.append(f"ipTM score: {metadata['iptm']:.4f}")

        if protein_info:
            seq = protein_info.get('sequence', '')
            if seq:
                desc_parts.append(f"Sequence length: {len(seq)}")
                desc_parts.append(f"Sequence preview: {seq[:50]}...")

        return "\n".join(desc_parts)

    def _format_complex_description(
        self,
        pdb_path: str,
        metadata: Dict,
        protein_info: Dict,
        antigen_id: str
    ) -> str:
        """Format output description for complex prediction."""
        desc_parts = ["Antigen-antibody complex structure prediction completed."]
        desc_parts.append(f"PDB file: {pdb_path}")
        desc_parts.append(f"Antigen chain ID: {antigen_id}")

        if metadata:
            if 'lddt_ca' in metadata:
                desc_parts.append(f"lDDT-Ca score: {metadata['lddt_ca']:.4f}")
            if 'ptm' in metadata:
                desc_parts.append(f"pTM score: {metadata['ptm']:.4f}")
            if 'iptm' in metadata:
                desc_parts.append(f"ipTM score: {metadata['iptm']:.4f}")

        if protein_info:
            seq = protein_info.get('sequence', '')
            if seq:
                desc_parts.append(f"Total sequence length: {len(seq)}")

        return "\n".join(desc_parts)

    def _format_epitope_description(
        self,
        result: Dict,
        antigen_id: str,
        distance_threshold: float
    ) -> str:
        """Format output description for epitope prediction."""
        desc_parts = ["Epitope determination completed."]
        desc_parts.append(f"Antigen chain ID: {antigen_id}")
        desc_parts.append(f"Distance threshold: {distance_threshold} Angstroms")
        desc_parts.append(f"Epitope residue count: {result.get('epitope_count', 0)}")

        epitope_residues = result.get('epitope_residues', [])
        if epitope_residues:
            # Show first 10 residues as preview
            preview = epitope_residues[:10]
            preview_str = ", ".join([f"[{r[0]}, {r[1]}, {r[2]}]" for r in preview])
            desc_parts.append(f"Epitope residues (first 10): {preview_str}")
            if len(epitope_residues) > 10:
                desc_parts.append(f"... and {len(epitope_residues) - 10} more residues")

        return "\n".join(desc_parts)


if __name__ == "__main__":
    # Test the tool
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tool = TFoldRequester()

    # Test health check
    print("Testing health check...")
    healthy = tool._health_check()
    print(f"tFold API healthy: {healthy}")

    # Test antibody prediction (if API is available)
    if healthy:
        print("\nTesting antibody prediction...")
        results, messages = tool.run(
            prediction_type="antibody",
            heavy_chain="EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK",
            light_chain="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
            output_name="test_antibody"
        )
        print(f"Results: {results}")
        print(f"Message:\n{messages[0]}")