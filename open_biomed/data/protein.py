from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from typing_extensions import Self

from datetime import datetime
import numpy as np
import os
import pickle
from rdkit import Chem
import re

from open_biomed.tools.base_tool import Tool
from open_biomed.data.text import Text
from open_biomed.utils.exception import ProteinConstructError

AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

AA_NAME_NUMBER = {
    k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
}

BACKBONE_NAMES = ["CA", "C", "N", "O"]

ptable = Chem.GetPeriodicTable()

def enumerate_pdb_lines(lines: list[str]) -> Iterator[Dict[str, Any]]:
    # Load and parse lines within .pdb lines 
    for line in lines:
        if line[0:6].strip() == 'ATOM':
            element_symb = line[76:78].strip().capitalize()
            if len(element_symb) == 0:
                element_symb = line[13:14]
            if element_symb == 'D':
                element_symb = 'H'
            yield {
                'line': line,
                'type': 'ATOM',
                'atom_id': int(line[6:11]),
                'atom_name': line[12:16].strip(),
                'res_name': line[17:20].strip(),
                'chain': line[21:22].strip(),
                'res_id': int(line[22:26]),
                'res_insert_id': line[26:27].strip(),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'occupancy': float(line[54:60]) if len(line) >= 60 else 1.0,
                'segment': line[72:76].strip() if len(line) >= 76 else "",
                'element_symb': element_symb,
                'charge': line[78:80].strip() if len(line) >= 80 else "",
            }
        elif line[0:6].strip() == 'HEADER':
            yield {
                'type': 'HEADER',
                'value': line[10:].strip()
            }
        elif line[0:6].strip() == 'ENDMDL':
            break   # Some PDBs have more than 1 model.

class Residue(dict):
    def __init__(self, name: str=None, atoms: list[int]=None, chain: str="A", segment: str="", res_id: int=1, res_insert_id: str="", chain_res_id: str=None) -> None:
        self.name = name
        self.atoms = atoms
        self.chain = chain
        self.segment = segment
        self.res_id = res_id
        self.res_insert_id = res_insert_id
        self.chain_res_id = chain_res_id

class Protein:
    def __init__(self) -> None:
        super().__init__()
        self.name = None

        # basic properties: 1D sequence, 3D coordinates
        self.sequence = None
        self.residues = None           # Backbone residues
        self.all_atom = None           # All-atom coordinates
        self.chi_angles = None         # Chi-angles of side chains

        # other related properties: textual descriptions and identifier in knowledge graph
        self.description = None
        self.kg_accession = None

    @classmethod
    def from_fasta(cls, fasta: str) -> Self:
        # initialize a protein with a fasta sequence
        protein = cls()
        protein.sequence = fasta
        protein._add_name()
        return protein

    @classmethod
    def from_fasta_file(cls, fasta_file: str) -> Self:
        # initialize a protein with a fasta file
        pass

    @classmethod
    def from_pdb(cls, pdb_lines: List[str], removeHs=True) -> Self:
        # TODO: handle multi-chain inputs 
        protein = cls()
        protein._add_name()
        protein.residues, protein.all_atom = [], []
        residues_tmp = {}
        for data in enumerate_pdb_lines(pdb_lines):
            if data['type'] == 'HEADER':
                protein.description = data['value'].lower()
                continue
            if removeHs and data['element_symb'] == 'H':
                continue
            if data['res_name'] not in AA_NAME_NUMBER:
                continue
            protein.all_atom.append({
                "line": data['line'],
                "pos": np.array([data['x'], data['y'], data['z']]),
                "atom_name": data['atom_name'],
                "occupancy": data['occupancy'],
                "atomic_number": ptable.GetAtomicNumber(data['element_symb']),
                "weight": ptable.GetAtomicWeight(data['element_symb']),
                "aa_type": AA_NAME_NUMBER[data['res_name']],
                "is_backbone": data['atom_name'] in BACKBONE_NAMES,
            })

            chain_res_id = '%s_%s_%d_%s' % (data['chain'], data['segment'], data['res_id'], data['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = Residue(**{
                    'name': data['res_name'],
                    'atoms': [len(protein.all_atom) - 1],
                    'chain': data['chain'],
                    'segment': data['segment'],
                    'res_id': data['res_id'],
                    'res_insert_id': data['res_insert_id'],
                    'chain_res_id': chain_res_id
                })
            else:
                assert residues_tmp[chain_res_id].name == data['res_name']
                assert residues_tmp[chain_res_id].chain == data['chain']
                residues_tmp[chain_res_id].atoms.append(len(protein.all_atom) - 1)
        
        protein.sequence = ""
        for residue in residues_tmp.values():
            protein.residues.append(residue)
            protein.sequence += AA_NAME_SYM[residue.name]
            sum_pos, sum_mass = np.zeros([3], dtype=np.float32), 0
            for atom_idx in residue.atoms:
                atom = protein.all_atom[atom_idx]
                sum_pos += atom["pos"] * atom["weight"]
                sum_mass += atom["weight"]
                if atom["atom_name"] in BACKBONE_NAMES:
                    protein.residues[-1].__setattr__('pos_%s' % atom["atom_name"], atom["pos"])
            protein.residues[-1].center_of_mass = sum_pos / sum_mass
        
        return protein

    @classmethod
    def from_pdb_file(cls, pdb_file: str, removeHs: bool=True) -> Self:
        # initialize a protein with a pdb file
        protein = Protein.from_pdb(open(pdb_file, "r").readlines(), removeHs=removeHs)
        protein.name = pdb_file.split("/")[-1].rstrip(".pdb")
        return protein

    @classmethod
    def from_binary_file(cls, file: str) -> Self:
        return pickle.load(open(file, "rb"))

    @staticmethod
    def folding(sequence: str, method: str='esmfold') -> np.ndarray:
        # Generate the 3D coordinates of the protein backbone
        pass

    @staticmethod
    def inverse_folding(conformer: np.ndarray, method: str='esm-1f') -> str:
        # Generate the amino acid sequence based on protein backbone coordinates
        pass

    @staticmethod
    def convert_chi_angles_to_coordinates(backbone: np.ndarray, chi_angles: np.ndarray) -> np.ndarray:
        # Calculate all-atom coordinates with backbone structure and chi-angles
        pass

    @staticmethod
    def convert_coordinates_to_chi_angles(coordinates: np.ndarray) -> np.ndarray:
        # Calculate chi-angles with all-atom coordinates
        pass

    def _add_name(self) -> None:
        if self.name is None:
            self.name = "protein_" + re.sub(r"[-:.]", "_", datetime.now().isoformat(sep="_", timespec="milliseconds"))

    def _add_sequence(self, base: str='backbone') -> None:
        # Add class property: sequence, based on backbone
        pass

    def _add_backbone(self, method: str='esmfold', base: str='sequence') -> None:
        # Add class property: backbone, based on sequence
        pass
    
    def _add_description(self, text_database: Dict[Any, Text], identifier_key: str='sequence', base: str='sequence') -> None:
        # Add class property: description, based on sequence
        pass

    def _add_kg_accession(self, kg_database: Dict[Any, str], identifier_key: str='sequence', base: str='sequence') -> None:
        # Add class property: kg_accession, based sequence
        pass

    def save_binary(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.pkl"

        if not os.path.exists(file) or overwrite:
            pickle.dump(self, open(file, "wb"))
        return file

    def save_pdb(self, file: Optional[str]=None, overwrite: bool=False, residue_indices: Optional[List[int]]=None) -> str:
        # Save the protein as a pdb file
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.pdb"

        if not os.path.exists(file) or overwrite:
            atom_cnt = 0
            with open(file, "w") as f:
                for i, residue in enumerate(self.residues):
                    if residue_indices is not None and i not in residue_indices:
                        continue
                    for atom_id in residue.atoms:
                        atom = self.all_atom[atom_id]
                        atom_cnt += 1
                        atom_name = f"  {atom['atom_name']:<3}" if len(atom['atom_name']) <= 3 else f" {atom['atom_name']:<4}"
                        f.write(f"ATOM  {atom_cnt:5}" + atom_name + f" {residue.name:3} {residue.chain:1}{residue.res_id:4}{residue.res_insert_id:1}   {atom['pos'][0]:8.3f}{atom['pos'][1]:8.3f}{atom['pos'][2]:8.3f}{atom.get('occupancy', 1.00):6.2f}{atom.get('temp_factor', 0.00):6.2f}           {atom['atom_name'][0]}\n")
        return file

    def __len__(self) -> int:
        if self.sequence is not None:
            return len(self.sequence)
        if self.residues is not None:
            return len(self.residues)
        raise AttributeError

    def __str__(self) -> str:
        self._add_sequence()
        return self.sequence

def protein_sequence_similarity(protein1: Protein, protein2: Protein) -> Tuple[float, str, str]:
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment

    seq1 = protein1.sequence
    seq2 = protein2.sequence
    alignments = pairwise2.align.globalms(seq1, seq2, 1, 0, 0, 0)
    alignment = alignments[0]
    # alignment: (aligned_seq1, aligned_seq2, score, begin, end)
    aligned_seq1, aligned_seq2, score, start, end = alignment
    # Calculate similarity as (number of matches) / (alignment length)
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2) if a != "-" and b != "-")
    alignment_length = max(len(aligned_seq1.replace("-", "")), len(aligned_seq2.replace("-", "")))
    if alignment_length == 0:
        similarity = 0.0
    else:
        similarity = matches / alignment_length
    output = format_alignment(*alignment)
    aligned_lines = output.splitlines()
    seq_line_1 = aligned_lines[0] if len(aligned_lines) > 0 else ""
    seq_line_2 = aligned_lines[2] if len(aligned_lines) > 2 else ""
    return similarity, seq_line_1, seq_line_2

