from typing import Tuple, List

import os
import subprocess
import glob
import logging
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.tools.base_tool import Tool


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ProteinBindingSitePrediction(Tool):
    def __init__(self, output_path: str = "./tmp/p2pocket") -> None:
        self.output_path = output_path
    
    def print_usage(self) -> str:
        return "\n".join([
            'Protein Binding Site Prediction',
            'Inputs: PDB file of the protein, or protein sequence (future support)',
            'Outputs: Multiple predicted binding sites'
        ])

    def run(self, protein: Protein, threads: int=8) -> Tuple[List[Pocket], List[str]]:
        pdb_file = protein.save_pdb()

        pdb_filename = os.path.basename(pdb_file)
        pdb_name = os.path.splitext(pdb_filename)[0]
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)   
        output_path = os.path.join(self.output_path, f"p2pocket_{pdb_name}")

        try:
            # Construct the command
            command = [
                "./third_party/p2rank_2.5/prank",
                "predict",
                "-f", pdb_file,
                "-threads", str(threads),
                "-o", output_path
            ]

            # Execute the command
            logging.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0:
                logging.info(f"Successfully processed {pdb_file}")
            else:
                logging.error(f"Failed to process {pdb_file}")
                logging.error(f"Error: {result.stderr}")

            
            file = output_path + "/" + pdb_name + ".pdb_predictions.csv"
            # 手动读取CSV，避免pandas版本兼容性问题
            with open(file, 'r') as f:
                lines = f.readlines()

            pocket_residues = []
            for line in lines[1:]:  # 跳过header
                values = line.strip().split(',')
                if len(values) >= 10:
                    residue_ids_str = values[9]  # residue_ids列
                    residue_ids = residue_ids_str.split()
                    parsed_residues = []
                    for i in residue_ids:
                        try:
                            parts = i.split("_")
                            if len(parts) >= 2:
                                # Keep (chain, res_id) pair for multi-chain proteins
                                parsed_residues.append((parts[0], int(parts[1])))
                        except (ValueError, IndexError):
                            pass
                    if parsed_residues:
                        pocket_residues.append(parsed_residues)


            protein = Protein.from_pdb_file(pdb_file)
            # 创建PDB编号到residue索引的映射，包含chain信息
            res_id_to_idx = {(res.chain, res.res_id): i for i, res in enumerate(protein.residues)}

            # 将PDB原始编号转换为protein.residues的索引
            pocket_indices = []
            for pdb_ids in pocket_residues:
                indices = []
                for chain, res_id in pdb_ids:
                    idx = res_id_to_idx.get((chain, res_id))
                    if idx is not None:
                        indices.append(idx)
                    else:
                        logging.warning(f"Residue {chain}_{res_id} not found in protein")
                if indices:
                    pocket_indices.append(indices)

            random.shuffle(pocket_indices)

            pockets, pocket_paths = [], []
            for pocket_idx in pocket_indices:
                try:
                    pocket = Pocket.from_protein_subseq(protein, pocket_idx)
                    pocket_path = pocket.save_binary()
                    pockets.append(pocket)
                    pocket_paths.append(pocket_path)
                except Exception as e:
                    logging.error(f"An error occurred: {str(e)}")
            return pockets, pocket_paths
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return [], []



if __name__ == "__main__":
    pdb_file = "third_party/p2rank_2.5/test_data/1fbl.pdb"
    pocket_predictor = ProteinBindingSitePrediction()
    pocket = pocket_predictor.run(pdb_file)
    print(pocket)


class ProdigyBindingAffinity(Tool):
    """
    Predict binding affinity for protein-protein complexes using PRODIGY.

    PRODIGY (PROtein binding affinity prediction using contact enerGY) predicts
    the binding affinity of protein-protein complexes based on intermolecular contacts.

    Reference: Vangone A, Bonvin AMJJ (2015) "Contacts-based prediction of binding
    affinity in protein-protein complexes"
    """

    def __init__(self, distance_cutoff: float = 5.5) -> None:
        self.distance_cutoff = distance_cutoff

    def print_usage(self) -> str:
        return "\n".join([
            'PRODIGY Binding Affinity Prediction',
            'Inputs: {"protein_complex": PDB file path}',
            'Outputs: float (predicted binding affinity in kcal.mol-1)',
            'Parameters: distance_cutoff (default: 5.5)'
        ])

    def run(self, protein_complex: str = "", distance_cutoff: float = None) -> Tuple[List[float], List[str]]:
        """
        Predict binding affinity for a protein complex.

        Args:
            protein_complex: PDB file path containing the protein complex
            distance_cutoff: Distance cutoff for calculating ICs (default: 5.5)

        Returns:
            Tuple of (binding affinity score list, description message list)
        """
        if distance_cutoff is None:
            distance_cutoff = self.distance_cutoff

        pdb_file = protein_complex

        if not os.path.exists(pdb_file):
            logging.error(f"PDB file not found: {pdb_file}")
            return [0.0], ["Error: PDB file not found"]

        try:
            command = [
                'prodigy',
                pdb_file,
                '--distance-cutoff', str(distance_cutoff)
            ]

            logging.info(f"Running PRODIGY: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logging.error(f"PRODIGY failed: {result.stderr}")
                return [0.0], [f"Error: {result.stderr}"]

            output = result.stdout
            # Parse PRODIGY output
            # Format: [++] Predicted binding affinity (kcal.mol-1):    -11.6
            if "Predicted binding affinity" in output:
                for line in output.split("\n"):
                    if "Predicted binding affinity (kcal.mol-1)" in line:
                        score = float(line.split(":")[-1].strip())
                        logging.info(f"Predicted binding affinity: {score} kcal.mol-1")
                        return [score], [f"Binding affinity: {score} kcal.mol-1 (distance_cutoff={distance_cutoff})"]

            logging.error(f"Failed to parse PRODIGY output")
            return [0.0], ["Error: Failed to parse output"]

        except subprocess.TimeoutExpired:
            logging.error("PRODIGY timed out")
            return [0.0], ["Error: PRODIGY timed out"]
        except FileNotFoundError:
            logging.error("PRODIGY not found. Install with: pip install prodigy-prot")
            return [0.0], ["Error: PRODIGY not installed"]
        except Exception as e:
            logging.error(f"PRODIGY error: {e}")
            return [0.0], [f"Error: {e}"]


class IgGMAntibodyDesign(Tool):
    """
    Antibody design using IgGM model.

    IgGM supports:
    - Epitope-conditioned de novo antibody design
    - Antibody affinity maturation
    - Epitope prediction from antigen-antibody complex

    Reference: Tencent AI4S IgGM (https://github.com/TencentAI4S/IgGM)
    """

    def __init__(self) -> None:
        self._model_loaded = False
        self._design_script = None

    def print_usage(self) -> str:
        return "\n".join([
            'IgGM Antibody Design',
            'Inputs:',
            '  - fasta: FASTA file path with design requirement (X for design region)',
            '  - antigen: Antigen PDB file path',
            '  - epitope: (optional) Epitope residue numbers, e.g. "7 8 9 10 11"',
            '  - fasta_origin: (optional) Original antibody FASTA for affinity maturation',
            '  - task: "design" (default) or "affinity_maturation"',
            '  - num_samples: Number of samples per residue (default: 10)',
            'Outputs: Designed antibody FASTA and PDB files'
        ])

    def _load_model(self):
        """Load IgGM model and check installation."""
        if self._design_script is None:
            # Check if IgGM is installed
            try:
                import IgGM
                self._design_script = IgGM.design
                logging.info("IgGM loaded successfully")
            except ImportError:
                # Try to find the design.py script
                iggm_paths = [
                    "./third_party/IgGM/design.py",
                    "./IgGM/design.py",
                    os.path.expanduser("~/IgGM/design.py")
                ]
                for path in iggm_paths:
                    if os.path.exists(path):
                        self._design_script = path
                        logging.info(f"Found IgGM design.py at {path}")
                        break
                if self._design_script is None:
                    logging.warning("IgGM not installed. Will use subprocess if available.")
        return self._design_script

    def run(
        self,
        fasta: str = "",
        antigen: str = "",
        epitope: str = "",
        fasta_origin: str = "",
        task: str = "design",
        num_samples: int = 10,
        output_path: str = None
    ) -> Tuple[List[str], List[str]]:
        """
        Design antibody using IgGM.

        Args:
            fasta: FASTA file path with design requirement (X marks design regions)
            antigen: Antigen PDB file path
            epitope: Epitope residue numbers (space-separated, optional for design)
            fasta_origin: Original antibody FASTA for affinity maturation
            task: "design" for de novo design, "affinity_maturation" for maturation
            num_samples: Number of samples per residue
            output_path: Output directory path

        Returns:
            Tuple of (output file paths list, description messages list)
        """
        import time
        import shutil

        # Validate inputs
        if not fasta:
            return [""], ["Error: FASTA file path is required"]
        if not antigen:
            return [""], ["Error: Antigen PDB file path is required"]

        # Generate output path
        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = f"./tmp/antibody_design_{timestamp}"

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Check if input files exist
        if not os.path.exists(fasta):
            return [""], [f"Error: FASTA file not found: {fasta}"]
        if not os.path.exists(antigen):
            return [""], [f"Error: Antigen PDB file not found: {antigen}"]

        try:
            # Try using IgGM Python package first
            design_script = self._load_model()

            if design_script is not None and not isinstance(design_script, str):
                # Use IgGM Python API
                logging.info(f"Running IgGM {task}...")

                if task == "design":
                    results = design_script(
                        fasta=fasta,
                        antigen=antigen,
                        epitope=epitope.split() if epitope else None,
                        output_dir=output_path,
                        num_samples=num_samples
                    )
                elif task == "affinity_maturation":
                    if not fasta_origin:
                        return [""], ["Error: fasta_origin is required for affinity maturation"]
                    results = design_script(
                        fasta=fasta,
                        antigen=antigen,
                        fasta_origin=fasta_origin,
                        run_task="affinity_maturation",
                        output_dir=output_path,
                        num_samples=num_samples
                    )
                else:
                    return [""], [f"Error: Unknown task '{task}'. Use 'design' or 'affinity_maturation'"]

                # Collect output files
                output_files = []
                for f in os.listdir(output_path):
                    if f.endswith('.pdb') or f.endswith('.fasta'):
                        output_files.append(os.path.join(output_path, f))

                return output_files, [f"Antibody design completed. Output saved to {output_path}"]

            else:
                # Use subprocess to run design.py script
                script_path = design_script
                if script_path is None or not os.path.exists(script_path):
                    # Check common installation paths
                    script_path = "./third_party/IgGM/design.py"
                    if not os.path.exists(script_path):
                        return [""], ["Error: IgGM not installed. Please install from https://github.com/TencentAI4S/IgGM"]

                # Build command
                cmd = ["python", script_path]
                cmd.extend(["--fasta", fasta])
                cmd.extend(["--antigen", antigen])
                cmd.extend(["--output", output_path])
                cmd.extend(["--num_samples", str(num_samples)])

                if epitope:
                    cmd.extend(["--epitope"] + epitope.split())

                if task == "affinity_maturation":
                    cmd.extend(["--run_task", "affinity_maturation"])
                    if fasta_origin:
                        cmd.extend(["--fasta_origin", fasta_origin])

                logging.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode != 0:
                    logging.error(f"IgGM error: {result.stderr}")
                    return [""], [f"Error: {result.stderr}"]

                # Collect output files
                output_files = []
                for f in os.listdir(output_path):
                    if f.endswith('.pdb') or f.endswith('.fasta'):
                        output_files.append(os.path.join(output_path, f))

                return output_files, [f"Antibody design completed. Output saved to {output_path}"]

        except subprocess.TimeoutExpired:
            logging.error("IgGM timeout")
            return [""], ["Error: IgGM execution timeout (max 600s)"]
        except Exception as e:
            logging.error(f"IgGM error: {e}")
            return [""], [f"Error: {str(e)}"]


class SimilarProteinSearch(Tool):
    """
    Search for similar proteins using FoldSeek (structure similarity).

    Note: MSA (sequence similarity) search is now handled via direct API calls
    documented in skills/similar-protein-retrieval/SKILL.md, not through this tool.

    Outputs:
    - FoldSeek: .m8 file with similar structure hits
    """

    def __init__(self) -> None:
        self._foldseek_requester = None

    def print_usage(self) -> str:
        return "\n".join([
            'Similar Protein Search (Structure)',
            'Inputs:',
            '  - protein: PDB file path',
            '  - database: (optional) List of databases for FoldSeek ["pdb100", "afdb50"]',
            'Outputs:',
            '  - FoldSeek: Path to result directory with .m8 file',
            'Note: MSA sequence search is handled via direct API calls (see SKILL.md)'
        ])

    def _load_foldseek_requester(self):
        """Load FoldSeek requester."""
        if self._foldseek_requester is None:
            from open_biomed.tools.web_request_tools import FoldSeekRequester
            self._foldseek_requester = FoldSeekRequester()
        return self._foldseek_requester

    def run(
        self,
        protein: str = "",
        database: List[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Search for similar protein structures using FoldSeek.

        Args:
            protein: PDB file path (must exist on server)
            database: List of FoldSeek databases (optional)

        Returns:
            Tuple of (result paths list, description messages list)
        """
        import asyncio

        if not protein:
            return [""], ["Error: protein input (PDB file path) is required"]

        # Check if file exists
        if not os.path.exists(protein):
            return [""], [f"Error: PDB file not found: {protein}"]

        try:
            # Load protein from PDB file
            from open_biomed.data import Protein
            protein_obj = Protein.from_pdb_file(protein)

            requester = self._load_foldseek_requester()

            # Override database if specified
            if database:
                requester.database = database

            logging.info(f"Running FoldSeek search with databases: {requester.database}")

            # Run async method
            result_paths, messages = asyncio.run(requester.run_async(protein_obj))

            return result_paths, [f"FoldSeek results saved to {messages[0]}"]

        except ImportError as e:
            logging.error(f"Import error: {e}")
            return [""], ["Error: Required libraries not installed"]
        except Exception as e:
            logging.error(f"Similar protein search error: {e}")
            return [""], [f"Error: {str(e)}"]