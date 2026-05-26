from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

import copy
from datetime import datetime
import gzip
import logging
import math
import numpy as np
import os
import pickle
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, BRICS,MACCSkeys, rdMolDescriptors, Descriptors, Lipinski
from rdkit.Chem.AllChem import RWMol
from rdkit.six import iteritems
from rdkit.six.moves import cPickle
import re

from open_biomed.tools.base_tool import Tool
from open_biomed.data.text import Text
from open_biomed.utils.exception import MoleculeConstructError

_fscores = None

class Molecule:
    def __init__(self) -> None:
        super().__init__()
        self.name = None

        # basic properties: 1D SMILES/SELFIES strings, RDKit mol object, 2D graphs, 3D coordinates
        self.smiles = None
        self.selfies = None
        self.rdmol = None
        self.graph = None
        self.conformer = None

        # other related properties: image, textual descriptions and identifier in knowledge graph
        self.img = None
        self.description = None
        self.kg_accession = None

    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        # initialize a molecule with a SMILES string
        molecule = cls()
        molecule.smiles = smiles
        molecule._add_rdmol(base="smiles")
        return molecule

    @classmethod
    def from_selfies(cls, selfies: str) -> Self:
        import selfies as sf
        molecule = cls()
        molecule.selfies = selfies
        molecule.smiles = sf.decoder(selfies)
        return molecule

    @classmethod
    def from_rdmol(cls, rdmol: RWMol) -> Self:
        # initialize a molecule with a RDKit molecule
        molecule = cls()
        molecule.rdmol = rdmol
        molecule.smiles = Chem.MolToSmiles(rdmol)
        conformer = rdmol.GetConformer()
        if conformer is not None:
            molecule.conformer = np.array(conformer.GetPositions())
        return molecule

    @classmethod
    def from_pdb(cls, pdb_lines: List[str]) -> Self:
        # initialize a molecule with pdb lines
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("/open_biomed/data", ""), "tmp", "molecule.pdb")
        with open(path, "w") as f:
            f.write("".join(pdb_lines))
        return cls.from_pdb_file(path)

    @classmethod
    def from_pdb_file(cls, pdb_file: str) -> Self:
        # initialize a molecule with a .pdb file
        rdmol = Chem.MolFromPDBFile(pdb_file)
        return cls.from_rdmol(rdmol)

    @classmethod
    def from_sdf_file(cls, sdf_file: str) -> Self:
        # initialize a molecule with a .sdf file
        loader = Chem.SDMolSupplier(sdf_file)
        for mol in loader:
            if mol is not None:
                molecule = Molecule.from_rdmol(mol)
                conformer = mol.GetConformer()
                molecule.conformer = np.array(conformer.GetPositions())
        molecule.name = sdf_file.split("/")[-1].strip(".sdf")
        return molecule

    @classmethod
    def from_pdbqt_file(cls, pdbqt_file: str) -> Self:
        # initialize a molecule with a .pdbqt file
        try:
            # import pdb; pdb.set_trace()
            from openbabel import pybel
            from openbabel import openbabel as ob
            sdf_file = pdbqt_file.replace(".pdbqt", ".sdf")
            mol = next(pybel.readfile("pdbqt", pdbqt_file))

            def fixup(mol):
                mol.OBMol.BeginModify()
                for atom in ob.OBMolAtomIter(mol.OBMol):
                    if atom.GetAtomicNum() == 7 and atom.IsInRing() and atom.GetExplicitValence() > 3:
                        # Find all rings in the molecule that contain this atom and set all atoms and bonds in those rings as aromatic
                        # ob.OBRingFinder does not exist. Instead, ensure ring perception is performed with OBMol.FindRingAtomsAndBonds()
                        mol.OBMol.FindRingAtomsAndBonds()
                        for ring in ob.OBMol.GetSSSR(mol.OBMol):  # SSSR expects an OBMol argument
                            if atom.GetIdx() in ring._path:
                                logging.info(f"Setting ring containing atom {atom.GetIdx()} as aromatic: {ring._path}")
                                for idx in ring._path:
                                    ring_atom = mol.OBMol.GetAtom(idx)
                                    ring_atom.SetAromatic(True)
                                for i in range(len(ring._path)):
                                    idx1 = ring._path[i]
                                    idx2 = ring._path[(i + 1) % len(ring._path)]
                                    bond = mol.OBMol.GetBond(mol.OBMol.GetAtom(idx1), mol.OBMol.GetAtom(idx2))
                                    if bond:
                                        bond.SetAromatic(True)
                                        bond.SetBondOrder(1)
                mol.OBMol.SetAromaticPerceived(True)
                mol.OBMol.EndModify()
                mol.OBMol.PerceiveBondOrders()
                return mol

            mol = fixup(mol)
            mol.OBMol.DeleteHydrogens()
            mol.write("sdf", sdf_file, overwrite=True)
            return cls.from_sdf_file(sdf_file)
        except ImportError:
            logging.warning("OpenBabel not installed. This function return None.")
            return None

    @classmethod
    def from_image_file(cls, image_file: str) -> Self:
        # initialize a molecule with a image file
        pass

    @classmethod
    def from_binary_file(cls, file: str) -> Self:
        return pickle.load(open(file, "rb"))

    @staticmethod
    def convert_smiles_to_rdmol(smiles: str, canonical: bool=True) -> RWMol:
        # Convert the smiles string into rdkit mol
        # If the smiles is invalid, raise MolConstructError
        pass

    @staticmethod
    def generate_conformer(rdmol: RWMol, method: str='mmff', num_conformers: int=1) -> np.ndarray:
        # Generate 3D conformer with algorithms in RDKit
        # TODO: identify if ML-based conformer generation can be applied
        pass

    def _add_name(self) -> None:
        if self.name is None:
            self.name = "mol_" + re.sub(r"[-:.]", "_", datetime.now().isoformat(sep="_", timespec="milliseconds"))

    def _add_smiles(self, base: str='rdmol') -> None:
        # Add class property: smiles, based on selfies / rdmol / graph, default: rdmol
        pass

    def _add_selfies(self, base: str='smiles') -> None:
        import selfies as sf
        # Add class property: selfies, based on smiles / selfies / rdmol / graph, default: smiles
        if base == "smiles":
            self.selfies = sf.encoder(self.smiles, strict=False)
        else:
            raise NotImplementedError

    def _add_rdmol(self, base: str='smiles') -> None:
        # Add class property: rdmol, based on smiles / selfies / graph, default: smiles
        if self.rdmol is not None:
            return
        if base == 'smiles':
            self.rdmol = Chem.MolFromSmiles(self.smiles)
        if self.conformer is not None:
            conf = mol_array_to_conformer(self.conformer)
            self.rdmol.AddConformer(conf)

    def _add_conformer(self, mode: str='2D', base: str='rdmol') -> None:
        # Add class property: conformer, based on smiles / selfies / rdmol, default: rdmol
        if self.conformer is None:
            self._add_rdmol()
            if mode == '2D':
                AllChem.Compute2DCoords(self.rdmol)
            elif mode == '3D':
                self.rdmol = Chem.AddHs(self.rdmol)
                AllChem.EmbedMolecule(self.rdmol)
                AllChem.MMFFOptimizeMolecule(self.rdmol)
                self.rdmol = Chem.RemoveHs(self.rdmol)
            conformer = self.rdmol.GetConformer()
            self.conformer = np.array(conformer.GetPositions())
    
    def _add_description(self, text_database: Dict[Any, Text], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: description, based on smiles / selfies / rdmol, default: smiles
        pass

    def _add_kg_accession(self, kg_database: Dict[Any, str], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: kg_accession, based on smiles / selfies / rdmol, default: smiles
        pass

    def save_sdf(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.sdf"

        if not os.path.exists(file) or overwrite:
            writer = Chem.SDWriter(file)
            self._add_rdmol()
            self._add_conformer()
            writer.write(self.rdmol)
        return file

    def save_binary(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.pkl"

        if not os.path.exists(file) or overwrite:
            pickle.dump(self, open(file, "wb"))
        return file

    def get_num_atoms(self) -> None:
        self._add_rdmol()
        return self.rdmol.GetNumAtoms()
    
    def calc_qed(self) -> float:
        try:
            from rdkit.Chem.QED import qed
            self._add_rdmol()
            return qed(self.rdmol)
        except Exception:
            return 0.0

    def calc_sa(self, normalize: bool=False) -> float:
        self._add_rdmol()
        sa = calc_sa_score(self.rdmol)
        if normalize:
            sa_norm = round((10 - sa) / 9, 2)
            return sa_norm
        else:
            return sa

    def calc_logp(self) -> float:
        from rdkit.Chem.Crippen import MolLogP
        self._add_rdmol()
        return MolLogP(self.rdmol)

    def calc_lipinski(self) -> float:
        try:
            self._add_rdmol()
            mol = copy.deepcopy(self.rdmol)
            Chem.SanitizeMol(mol)
            rule_1 = Descriptors.ExactMolWt(mol) < 500
            rule_2 = Lipinski.NumHDonors(mol) <= 5
            rule_3 = Lipinski.NumHAcceptors(mol) <= 10
            logp = self.calc_logp()
            rule_4 = (logp >= -2) & (logp <= 5)
            #rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
            #return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
            return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4]])
        except Exception:
            return 0.0

    def calc_distance(self) -> float:
        self._add_conformer()
        pdist = self.conformer[None, :] - self.conformer[:, None]
        return np.sqrt(np.sum(pdist ** 2, axis=-1))

    def __str__(self) -> str:
        return self.smiles

def molecule_fingerprint_similarity(mol1: Molecule, mol2: Molecule, fingerprint_type: str="morgan") -> float:
    # Calculate the fingerprint similarity of two molecules
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        if fingerprint_type == "morgan":
            fp1 = AllChem.GetMorganFingerprint(mol1.rdmol, 2)
            fp2 = AllChem.GetMorganFingerprint(mol2.rdmol, 2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        if fingerprint_type == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1.rdmol)
            fp2 = Chem.RDKFingerprint(mol2.rdmol)
        if fingerprint_type == "maccs":
            fp1 = MACCSkeys.GenMACCSKeys(mol1.rdmol)
            fp2 = MACCSkeys.GenMACCSKeys(mol2.rdmol)
        return DataStructs.FingerprintSimilarity(
            fp1, fp2,
            metric=DataStructs.TanimotoSimilarity
        )
    except Exception:
        return 0.0

def check_identical_molecules(mol1: Molecule, mol2: Molecule) -> bool:
    # Check if the two molecules are the same
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        return Chem.MolToInchi(mol1.rdmol) == Chem.MolToInchi(mol2.rdmol)
    except Exception:
        return False

def mol_array_to_conformer(conf: np.ndarray) -> Chem.Conformer:
    new_conf = Chem.Conformer(conf.shape[0])
    for i in range(conf.shape[0]):
        new_conf.SetAtomPosition(i, tuple(conf[i]))
    return new_conf

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
def calc_sa_score(molecule: Chem.RWMol) -> float:
    global _fscores
    if _fscores is None:
        fpscores_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "molecule", "fpscores.pkl.gz")
        data = cPickle.load(gzip.open(fpscores_file, "rb"))
        _fscores = {}
        for i in data:
            for j in range(1, len(i)):
                _fscores[i[j]] = float(i[0])

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(molecule, 2)  #<- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = molecule.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    ri = molecule.GetRingInfo()
    nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(molecule)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def calc_mol_diversity(mols: List[Molecule]) -> float:
    # Calculate the diversity of a list of molecules
    # Use the fingerprint similarity to calculate the diversity
    # The diversity is the average of the fingerprint similarity of all pairs of molecules
    dists = []
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            mol1 = mols[i]
            mol2 = mols[j]
            mol1._add_rdmol()
            mol2._add_rdmol()
            dists.append(1 - molecule_fingerprint_similarity(mol1, mol2, fingerprint_type="rdkit"))
    return np.mean(dists)

def calc_mol_rmsd(mol1: Molecule, mol2: Molecule) -> float:
    # Calculate the RMSD of two molecules
    try:
        if mol1.conformer is None or mol2.conformer is None:
            raise ValueError("Conformer is not available for RMSD calculation")
        assert mol1.get_num_atoms() == mol2.get_num_atoms(), "The number of atoms of two molecules must be the same!"
        mol1._add_rdmol()
        mol2._add_rdmol()
        return Chem.rdMolAlign.CalcRMS(mol1.rdmol, mol2.rdmol, maxMatches=30000)
    except Exception:
        return 1e4

def calc_mol_reasonable(mol: Molecule) -> Tuple[bool, float]:
    # Calculate if a molecule is reasonable (https://arxiv.org/abs/2503.01376)
    groups = []
    ring_info = mol.rdmol.GetRingInfo()
    ring_atoms = ring_info.AtomRings()
    group_indices = [-1] * len(ring_atoms)
    for ring_idx in range(len(ring_atoms)):
        if group_indices[ring_idx] == -1:
            group_indices[ring_idx] = len(groups)
            groups.append([ring_atoms[ring_idx]])
        stack = [ring_idx]
        while len(stack) > 0:
            ring_idx = stack.pop()
            for ring_idx2 in range(len(ring_atoms)):
                if group_indices[ring_idx2] == -1 and set(ring_atoms[ring_idx]).intersection(set(ring_atoms[ring_idx2])):
                    group_indices[ring_idx2] = group_indices[ring_idx]
                    stack.append(ring_idx2)
                    groups[group_indices[ring_idx]].append(ring_atoms[ring_idx2])
    reasonable = True
    unreasonable_atom_count = 0
    for group in groups:
        if len(group) > 1:
            no_sp2 = True
            for ring in group:
                for atom_idx in ring:
                    atom = mol.rdmol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                        no_sp2 = False
                        break
                if not no_sp2:
                    break
            if no_sp2:
                reasonable = False
                unreasonable_atom_count += sum(len(ring) for ring in group)
    if not reasonable:
        return False, unreasonable_atom_count / mol.get_num_atoms()
    
    co_cn_ids = set()
    for bond in mol.rdmol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if atom1.GetSymbol() == 'C' and (atom2.GetSymbol() == 'N' or atom2.GetSymbol() == 'O'):
                co_cn_ids.add(atom1.GetIdx())
            elif atom2.GetSymbol() == 'C' and (atom1.GetSymbol() == 'N' or atom1.GetSymbol() == 'O'):
                co_cn_ids.add(atom2.GetIdx())
   
    for group in groups:
        verified_sp2 = set()
        verified_non_sp2 = set()
        remaining_rings = group
        while len(remaining_rings) > 0:
            new_sp2, new_non_sp2 = set(), set()
            for ring in remaining_rings:
                remaining_atoms = set(ring) - verified_sp2 - verified_non_sp2
                is_sp2 = all(
                    mol.rdmol.GetAtomWithIdx(atom_idx).GetHybridization() == Chem.rdchem.HybridizationType.SP2
                    for atom_idx in remaining_atoms if atom_idx not in co_cn_ids and mol.rdmol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C'
                )
                is_non_sp2 = all(
                    mol.rdmol.GetAtomWithIdx(atom_idx).GetHybridization() != Chem.rdchem.HybridizationType.SP2
                    for atom_idx in remaining_atoms if atom_idx not in co_cn_ids and mol.rdmol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C'
                )
                if is_sp2:
                    new_sp2.update(remaining_atoms)
                    remaining_rings.remove(ring)
                elif is_non_sp2:
                    new_non_sp2.update(remaining_atoms)
                    remaining_rings.remove(ring)
            if len(new_sp2) == 0 and len(new_non_sp2) == 0:
                break

            verified_sp2.update(new_sp2)
            verified_non_sp2.update(new_non_sp2)
            
        if remaining_rings:
            reasonable = False
            unreasonable_atom_count += sum(len(ring) for ring in remaining_rings)
    return reasonable, unreasonable_atom_count / mol.get_num_atoms()

def calc_mol_fragment(mol: Molecule) -> List[int]:
    # Calculate the fragment indices for each atom in a molecule
    res = []
    for bond in BRICS.FindBRICSBonds(mol.rdmol):
        res.append(bond)
    cut_bonds = [list(ele[0]) for ele in res]
    cut_graph = [[] for _ in range(mol.get_num_atoms())]
    for bond in mol.rdmol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if [i, j] not in cut_bonds and [j, i] not in cut_bonds:
            cut_graph[i].append(j)
            cut_graph[j].append(i)
    
    frag_idx = [-1] * mol.get_num_atoms()
    cur_frag_idx = 0

    def dfs(u: int):
        frag_idx[u] = cur_frag_idx
        for v in cut_graph[u]:
            if frag_idx[v] == -1:
                dfs(v)

    for i in range(mol.get_num_atoms()):
        if frag_idx[i] == -1:
            dfs(i)
            cur_frag_idx += 1
    return frag_idx

class MoleculeQEDTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Calculate the drug-likeness (QED score) of a molecule
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: float (the QED score of the molecule)
"""

    def run(self, molecule: Molecule) -> float:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        scores, messages = [], []
        for mol in molecule:
            scores.append(mol.calc_qed())
            messages.append(f"The molecule has a QED score of {scores[-1]}")
        return scores, messages

class MoleculeSATool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Calculate the synthetic accessibility (SA score) of a molecule
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object), "normalize": bool (whether to normalize the SA score to range [0, 1], default: False)}
Outputs: float (the SA score of the molecule)
"""

    def run(self, molecule: Union[Molecule, List[Molecule]], normalize: bool=False) -> float:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        scores, messages = [], []
        for mol in molecule:
            scores.append(mol.calc_sa(normalize))
            messages.append(f"The molecule has a SA score of {scores[-1]}")
        return scores, messages

class MoleculeLogPTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Calculate the solubility (LogP score) of a molecule
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: float (the LogP score of the molecule)
"""

    def run(self, molecule: Union[Molecule, List[Molecule]]) -> float:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        scores, messages = [], []
        for mol in molecule:
            scores.append(mol.calc_logp())
            messages.append(f"The molecule has a LogP score of {scores[-1]}")
        return scores, messages

class MoleculeLipinskiTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Calculate the number of lipinski rules that a molecule satisfies
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: float (the number of lipinski rules that the molecule satisfies)
"""

    def run(self, molecule: Union[Molecule, List[Molecule]]) -> float:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        scores, messages = [], []
        for mol in molecule:
            scores.append(mol.calc_lipinski())
            messages.append(f"The molecule satisfies {scores[-1]} lipinski rules")
        return scores, messages

class MoleculePropertyCalculationTool:
    def __init__(self):
        """
        Initialize the calculator with a mapping of property names to their corresponding tool classes.
        """
        # Map property names to their corresponding tool classes
        self.tool_map = {
            "QED": MoleculeQEDTool,
            "SA": MoleculeSATool,
            "LogP": MoleculeLogPTool,
            "Lipinski": MoleculeLipinskiTool,
        }

    def run(self, molecule: Molecule, property: str) -> float:
        """
        Calculate the specified property for the given molecule using the appropriate tool.

        :param molecule: The molecule object to calculate the property for.
        :param property: The name of the property to calculate (e.g., "QED", "SA", "LogP", "Lipinski").
        :return: The calculated property value (converted to Python native type).
        """
        if property not in self.tool_map:
            raise ValueError(f"Unknown property: {property}")

        tool_class = self.tool_map[property]
        tool_instance = tool_class()
        scores, messages = tool_instance.run(molecule)
        # Convert numpy types to Python native types for FastAPI serialization
        score = scores[0]
        if property == "Lipinski":
            return int(score)
        else:
            return float(score)

class MoleculeSimilarityTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Calculate the Morgan fingerprint similarity of two molecules
Inputs: {"molecule_1": Molecule (an OpenBioMed Molecule object), "molecule_2": Molecule (an OpenBioMed Molecule object)}
Outputs: float (the Morgan fingerprint similarity of the two molecules)
"""

    def run(self, molecule_1: Union[Molecule, List[Molecule]], molecule_2: Union[Molecule, List[Molecule]]) -> Tuple[List[float], List[str]]:
        if isinstance(molecule_1, Molecule):
            molecule_1 = [molecule_1]
        if isinstance(molecule_2, Molecule):
            molecule_2 = [molecule_2]
        scores, messages = [], []
        for idx, mol1 in enumerate(molecule_1):
            mol2 = molecule_2[idx]
            scores.append(molecule_fingerprint_similarity(mol1, mol2, fingerprint_type="morgan"))
            messages.append(f"The Morgan fingerprint similarity of the two molecules is {scores[-1]}")
        return scores, messages
        

class MoleculeReasonableTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Calculate if a molecule is reasonable
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: Tuple[bool, float] (whether the molecule is reasonable, the percentage of unreasonable atoms)
"""

    def run(self, molecule: Union[Molecule, List[Molecule]]) -> Tuple[List[bool], List[str]]:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        scores, messages = [], []
        for mol in molecule:
            scores.append(calc_mol_reasonable(mol))
            messages.append(f"The molecule is reasonable: {scores[-1][0]}, the percentage of unreasonable atoms is {scores[-1][1]}")
        return scores, messages


class MoleculeFragmentTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Obtain the BRICS fragment of a molecule
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: List[str] (the SMILES strings of the BRICS fragments)
"""

    def run(self, molecule: Union[Molecule, List[Molecule]]) -> Tuple[List[str], List[str]]:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        frag_smiles, messages = [], []
        for mol in molecule:
            atom2frag = np.array(calc_mol_fragment(mol))
            for i in range(np.max(frag_idx) + 1):
                frag_idx = np.where(atom2frag == i)[0]
                frag_smiles.append(Chem.MolFragmentToSmiles(mol.rdmol, frag_idx.tolist(), kekuleSmiles=True))
            frag_smiles.append(calc_mol_fragment(mol))
            messages.append(f"The BRICS fragments of the molecule are {frag_smiles[-1]}")
        return frag_smiles, messages