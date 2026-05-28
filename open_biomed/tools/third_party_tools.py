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
                                parsed_residues.append(int(parts[1]))
                        except (ValueError, IndexError):
                            pass
                    if parsed_residues:
                        pocket_residues.append(parsed_residues)

            
            protein = Protein.from_pdb_file(pdb_file)
            # 创建PDB编号到residue索引的映射
            res_id_to_idx = {res.res_id: i for i, res in enumerate(protein.residues)}

            # 将PDB原始编号转换为protein.residues的索引
            pocket_indices = []
            for pdb_ids in pocket_residues:
                indices = []
                for pdb_id in pdb_ids:
                    idx = res_id_to_idx.get(pdb_id)
                    if idx is not None:
                        indices.append(idx)
                    else:
                        logging.warning(f"Residue {pdb_id} not found in protein")
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