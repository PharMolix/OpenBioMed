import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import pickle
import numpy as np
import torch

import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdmolops
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from transformers import BertTokenizer, T5Tokenizer

from feat.base_featurizer import BaseFeaturizer
from feat.kg_featurizer import SUPPORTED_KG_FEATURIZER
from feat.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from utils import to_clu_sparse

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

# Atom featurization: Borrowed from dgllife.utils.featurizers.py

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.
    """
    return [atom.GetFormalCharge()]

def atom_num_radical_electrons(atom):
    return [atom.GetNumRadicalElectrons()]

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.
    """
    return [atom.GetIsAromatic()]

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.
    """
    return [atom.IsInRing()]

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.
    """
    if not atom.HasProp('_CIPCode'):
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)

# Atom featurization: Borrowed from dgllife.utils.featurizers.py

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

class DrugOneHotFeaturizer(BaseFeaturizer):
    smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

    def __init__(self, config):
        super(DrugOneHotFeaturizer, self).__init__()
        self.max_len = config["max_len"]
        self.enc = OneHotEncoder().fit(np.array(self.smiles_char).reshape(-1, 1))

    def __call__(self, data):
        temp = [c if c in self.smiles_char else '?' for c in data]
        if len(temp) < self.max_len:
            temp = temp + ['?'] * (self.max_len - len(temp))
        else:
            temp = temp[:self.max_len]
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)

class DrugTransformerTokFeaturizer(BaseFeaturizer):
    name2tokenizer = {
        "bert": BertTokenizer,
        "t5": T5Tokenizer
    }

    def __init__(self, config):
        super(DrugTransformerTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.tokenizer = self.name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)

    def __call__(self, data):
        result = self.tokenizer(data, max_length=self.max_length, padding=True, truncation=True)
        return result

class DrugBPEFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugBPEFeaturizer, self).__init__()

        from subword_nmt.apply_bpe import BPE, read_vocabulary
        import codecs

        self.bpe = BPE(
            codecs.open(config["code_name"], encoding="utf8"), 
            vocab=read_vocabulary(codecs.open(config["vocabulary"], encoding="utf8"), config["vocabulary_threshold"]),
            separator="~~", 
        )
        self.vocabs = {}
        lines = open(config["vocabulary"], "r").readlines()
        for line in lines:
            wd = line.strip('\n').split(' ')
            self.vocabs[wd[0]] = len(self.vocabs)
        self.max_length = config["max_length"]

    def _preprocess_smiles(self, data):
        data = data.replace('(', '')
        data = data.replace(')', '')
        for i in range(10):
            item = str(i)
            data = data.replace(item, '')
        return data

    def __call__(self, data):
        data = self._preprocess_smiles(data)
        bpe_result = self.bpe.process_line(data).split(" ")
        result = [self.vocabs[x] if x in self.vocabs else len(self.vocabs) for x in bpe_result]
        if len(result) > self.max_length - 2:
            result = result[:self.max_length - 2]
        input_ids = torch.LongTensor([102] + [i + 30700 for i in result] + [103] + [0] * (self.max_length - 2 - len(result)))
        attn_mask = torch.LongTensor([1] * (len(result) + 2) + [0] * (self.max_length - len(result) - 2))
        token_type_ids = torch.zeros_like(attn_mask).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids
        }

class DrugFPFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugFPFeaturizer, self).__init__()
        self.config = config

    def __call__(self, data):
        mol = Chem.MolFromSmiles(data)
        if mol is not None:
            fp = Chem.RDKFingerprint(mol, fpSize=self.config["fpsize"])
            np_fp = np.zeros(self.config["fpsize"])
            DataStructs.ConvertToNumpyArray(fp, np_fp)
            if self.config["return_type"] == "pt":
                return torch.tensor(np_fp)
            else:
                return np_fp
        else:
            return None

class DrugTGSAFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugTGSAFeaturizer, self).__init__()
        self.config = config

    def atom_feature(self, atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        8 features are canonical, 2 features are from OGB
        """
        featurizer_funcs = [
            atom_type_one_hot,
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
            atom_is_in_ring,
            atom_chirality_type_one_hot,
        ]
        atom_feature = np.concatenate([func(atom) for func in featurizer_funcs], axis=0)
        return atom_feature

    def bond_feature(self, bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        featurizer_funcs = [bond_type_one_hot]
        bond_feature = np.concatenate([func(bond) for func in featurizer_funcs], axis=0)

        return bond_feature
    
    def __call__(self, data):
        mol = Chem.MolFromSmiles(data)
        """
        Converts SMILES string to graph Data object without remove salt
        :input: SMILES string (str)
        :return: pyg Data object
        """
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_feature(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = self.bond_feature(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )
        return graph

class DrugGraphFeaturizer(BaseFeaturizer):
    allowable_features = {
        'possible_atomic_num_list':       list(range(1, 119)) + ['misc'],
        'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_chirality_list':        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list':    [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            'misc'
        ],
        'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_is_aromatic_list':      [False, True],
        'possible_is_in_ring_list':       [False, True],
        'possible_bond_type_list':                 [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            'misc'
        ],
        'possible_bond_dirs':             [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ],
        'possible_bond_stereo_list':      [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ], 
        'possible_is_conjugated_list': [False, True]
    }

    def __init__(self, config):
        super(DrugGraphFeaturizer, self).__init__()
        self.config = config

    def __call__(self, data):
        if isinstance(data, str):
            mol = Chem.MolFromSmiles(data)
            # mol = AllChem.MolFromSmiles(data)
        else:
            mol = data
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            if self.config["name"] == "ogb":
                atom_feature = [
                    safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                    self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
                    safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                    safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                    safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                    safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                    safe_index(self.allowable_features['possible_hybridization_list'], atom.GetHybridization()),
                    self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                    self.allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
                ]
            else:
                atom_feature = [
                    safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                    self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
                ]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        if len(mol.GetBonds()) <= 0:  # mol has no bonds
            num_bond_features = 3 if self.config["name"] == "ogb" else 2
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if self.config["name"] == "ogb":
                    edge_feature = [
                        safe_index(self.allowable_features['possible_bond_type_list'], bond.GetBondType()),
                        self.allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                        self.allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                    ]
                else:
                    edge_feature = [
                        self.allowable_features['possible_bond_type_list'].index(bond.GetBondType()),
                        self.allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
                    ]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

class DrugGGNNFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugGGNNFeaturizer, self).__init__()
        self.max_n_atoms = config["max_n_atoms"]
        self.atomic_num_list = config["atomic_num_list"]
        self.bond_type_list = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            'misc'
        ]

    def __call__(self, data):
        if isinstance(data, str):
            mol = Chem.MolFromSmiles(data)
        else:
            mol = data
        Chem.Kekulize(mol)
        x = self._construct_atomic_number_array(mol, self.max_n_atoms)
        adj = self._construct_adj_matrix(mol, self.max_n_atoms)
        return x, adj, self._rescale_adj(adj) 

    def _construct_atomic_number_array(self, mol, out_size=-1):
        """Returns atomic numbers of atoms consisting a molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of returned array.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the tail of
                the array is padded with zeros.

        Returns:
            torch.tensor: a tensor consisting of atomic numbers
                of atoms in the molecule.
        """

        atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if len(atom_list) > self.max_n_atoms:
            atom_list =  atom_list[:self.max_n_atoms]

        if out_size < 0:
            result = torch.zeros(len(atom_list), len(self.atomic_num_list))
        else:
            result = torch.zeros(out_size, len(self.atomic_num_list))
        for i, atom in enumerate(atom_list):
            result[i, safe_index(self.atomic_num_list, atom)] = 1
        for i in range(len(atom_list), self.max_n_atoms):
            result[i, -1] = 1
        return result

    def _construct_adj_matrix(self, mol, out_size=-1, self_connection=True):
        """Returns the adjacent matrix of the given molecule.

        This function returns the adjacent matrix of the given molecule.
        Contrary to the specification of
        :func:`rdkit.Chem.rdmolops.GetAdjacencyMatrix`,
        The diagonal entries of the returned matrix are all-one.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of the returned matrix.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the adjacent
                matrix is expanded and zeros are padded to right
                columns and bottom rows.
            self_connection (bool): Add self connection or not.
                If True, diagonal element of adjacency matrix is filled with 1.

        Returns:
            adj (torch.tensor): The adjacent matrix of the input molecule.
                It is 2-dimensional tensor with shape (atoms1, atoms2), where
                atoms1 & atoms2 represent from and to of the edge respectively.
                If ``out_size`` is non-negative, the returned
                its size is equal to that value. Otherwise,
                it is equal to the number of atoms in the the molecule.
        """

        if out_size < 0:
            adj = torch.zeros(4, mol.GetNumAtoms(), mol.GetNumAtoms())
        else:
            adj = torch.zeros(4, out_size, out_size)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[safe_index(self.bond_type_list, bond.GetBondType()), i, j] = 1
            adj[safe_index(self.bond_type_list, bond.GetBondType()), j, i] = 1
        adj[3] = 1 - torch.sum(adj[:3], dim=0)
        return adj

    def _rescale_adj(self, adj):
        # Previous paper didn't use rescale_adj.
        # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
        # In this implementation, the normaliztion term is different
        # raise NotImplementedError
        # (256,4,9, 9):
        # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
        # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
        # usually first 3 matrices have no diagnal, the last has.
        # A_prime = self.A + sp.eye(self.A.shape[0])
        num_neighbors = adj.sum(dim=(0, 1)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[None, None, :] * adj
        return adj_prime

class DrugMGNNFeaturizer(BaseFeaturizer):
    allowable_atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    allowable_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_num_hs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_implicit_valence_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_hybridization_list = [
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 
        'other'
    ]
    allowable_cip_code_list = ['R', 'S']

    def __init__(self, config):
        super(DrugMGNNFeaturizer, self).__init__()
        self.config = config

    def __call__(self, data):
        if isinstance(data, str):
            mol = Chem.MolFromSmiles(data)
        else:
            mol = data
        
        atom_features_list = []
        for atom in mol.GetAtoms():
            encoding = self.one_of_k_encoding_unk(atom.GetSymbol(), self.allowable_atom_list)
            encoding += self.one_of_k_encoding(atom.GetDegree(), self.allowable_degree_list)
            encoding += self.one_of_k_encoding_unk(atom.GetTotalNumHs(), self.allowable_num_hs_list)
            encoding += self.one_of_k_encoding_unk(atom.GetImplicitValence(), self.allowable_implicit_valence_list)
            encoding += self.one_of_k_encoding_unk(atom.GetHybridization(), self.allowable_hybridization_list)
            encoding += [atom.GetIsAromatic()]
            try:
                encoding += self.one_of_k_encoding_unk(atom.GetProp("_CIPNode"), self.allowable_cip_code_list)
            except:
                encoding += [0, 0]
            encoding += [atom.HasProp("_ChiralityPossible")]
            encoding /= np.sum(encoding)
            atom_features_list.append(encoding)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

        if len(mol.GetBonds()) <= 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edges_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges_list.append((i, j))
                edges_list.append((j, i))
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

class DrugMultiScaleFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugMultiScaleFeaturizer, self).__init__()
        self.scales = config["scales"]
        self.featurizers = {}
        for scale in config["scales"]:
            conf = config[scale]
            self.featurizers[scale] = SUPPORTED_SINGLE_SCALE_DRUG_FEATURIZER[conf["name"]](conf)

    def __call__(self, data):
        feat = {}
        for scale in self.scales:
            feat[scale] = self.featurizers[scale](data)
        return feat


# same with graphmvp   
class DrugGraphFeaturizerV2(BaseFeaturizer):
    allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_is_aromatic_list':      [False, True],
    'possible_is_in_ring_list':       [False, True],
    'possible_bond_type_list':        [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_stereo_list':      [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ], 
    'possible_is_conjugated_list': [False, True]
    }
    def __init__(self, config):
        super(DrugGraphFeaturizerV2, self).__init__()
        self.config = config

    def __call__(self, data):
        if isinstance(data, str):
            mol = AllChem.MolFromSmiles(data)
        else:
            mol = data
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            if self.config["name"] == "ogb":
                atom_feature = [
                    safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                    self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
                    safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                    safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                    safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                    safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                    safe_index(self.allowable_features['possible_hybridization_list'], atom.GetHybridization()),
                    self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                    self.allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
                ]
            else:
                """
                atom_feature = [
                    safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                    self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
                ]
                """
                atom_feature = [self.allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        if len(mol.GetBonds()) <= 0:  # mol has no bonds
            num_bond_features = 3 if self.config["name"] == "ogb" else 2
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if self.config["name"] == "ogb":
                    edge_feature = [
                        safe_index(self.allowable_features['possible_bond_type_list'], bond.GetBondType()),
                        self.allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                        self.allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                    ]
                else:
                    """
                    edge_feature = [
                        self.allowable_features['possible_bond_type_list'].index(bond.GetBondType()),
                        self.allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
                    ]
                    """
                    edge_feature = [self.allowable_features['possible_bond_type_list'].index(bond.GetBondType())] + \
                           [self.allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

class DrugMultiModalFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(DrugMultiModalFeaturizer, self).__init__()
        self.modality = config["modality"]
        self.featurizers = {}
        if "structure" in config["modality"]:
            conf = config["featurizer"]["structure"]
            self.featurizers["structure"] = SUPPORTED_SINGLE_MODAL_DRUG_FEATURIZER[conf["name"]](conf)
        if "kg" in config["modality"]:
            conf = config["featurizer"]["kg"]
            self.featurizers["kg"] = SUPPORTED_KG_FEATURIZER[conf["name"]](conf)
        if "text" in config["modality"]:
            conf = config["featurizer"]["text"]
            self.featurizers["text"] = SUPPORTED_TEXT_FEATURIZER[conf["name"]](conf)

    def set_drug2kgid_dict(self, drug2kgid):
        self.featurizers["kg"].set_transform(drug2kgid)

    def set_drug2text_dict(self, drug2text):
        self.featurizers["text"].set_transform(drug2text)

    def __call__(self, data, skip=[]):
        feat = {}
        for modality in self.featurizers.keys():
            if modality not in skip:
                feat[modality] = self.featurizers[modality](data)
        return feat

    def __getitem__(self, index):
        if index not in self.modality:
            logger.error("%s is not a valid modality!" % (index))
            return None
        return self.featurizers[index]


SUPPORTED_SINGLE_SCALE_DRUG_FEATURIZER = {
    "OneHot": DrugOneHotFeaturizer,
    "KV-PLM*": DrugBPEFeaturizer,
    "transformer": DrugTransformerTokFeaturizer,
    "fp": DrugFPFeaturizer,
    "TGSA": DrugTGSAFeaturizer,
    "ogb": DrugGraphFeaturizer,
    "MGNN": DrugMGNNFeaturizer,
    "BaseGNN": DrugGraphFeaturizer,
    "BaseGNNv2": DrugGraphFeaturizerV2,
}

SUPPORTED_SINGLE_MODAL_DRUG_FEATURIZER = copy.deepcopy(SUPPORTED_SINGLE_SCALE_DRUG_FEATURIZER)
SUPPORTED_SINGLE_MODAL_DRUG_FEATURIZER["MultiScale"] = DrugMultiScaleFeaturizer

SUPPORTED_DRUG_FEATURIZER = copy.deepcopy(SUPPORTED_SINGLE_MODAL_DRUG_FEATURIZER)
SUPPORTED_DRUG_FEATURIZER["MultiModal"] = DrugMultiModalFeaturizer

def add_arguments(parser):
    parser.add_argument("--mode", type=str, choices=["unit_test", "interactive", "file"])
    parser.add_argument("--featurizer", type=str, default="fp")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--smiles_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--post_transform", type=str, default="")

    return parser

def unit_test():
    smi = "CCC=O"
    data = DrugGraphFeaturizer({"name": "ogb"})(smi)
    print(data.x, data.edge_index, data.edge_attr)

    smi = "C(C(C(=O)O)N)C(C(=O)O)O"
    data = DrugBPEFeaturizer({
        "name": "KV-PLM*",
        "code_name": "../assets/KV-PLM*/bpe_coding.txt",
        "vocabulary": "../assets/KV-PLM*/bpe_vocab.txt",
        "vocabulary_threshold": 80,
        "max_length": 32
    })(smi)
    print(data)

    smi = "OC(=O)C1=CC(=CC=C1O)\\N=N\\C1=CC=C(C=C1)S(=O)(=O)NC1=NC=CC=C1"
    data = DrugMGNNFeaturizer({})(smi)
    print(data.x, data.edge_index)

def featurize_file(args):
    with open(args.smiles_file, "r") as f:
        smis = [line.rstrip("\n") for line in f.readlines()]
    config = json.load(open(args.config_file, "r"))
    featurizer = SUPPORTED_DRUG_FEATURIZER[args.featurizer](config)
    result = [featurizer(smi) for smi in smis]
    if args.post_transform == "to_clu":
        result = to_clu_sparse(np.array(result))
        with open(args.output_file, "w") as f:
            f.write(result)
    else:
        pickle.dump(result, open(args.output_file, "wb"))

def run_featurize(args):
    # TODO: implement command line tool for featurizing SMILES
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    if args.mode == "unit_test":
        unit_test()
    elif args.mode == "file":
        featurize_file(args)
    elif args.mode == "interactive":
        run_featurize(args)