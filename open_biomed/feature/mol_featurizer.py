import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import copy
import json
import numpy as np
import pickle

import torch

import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdmolops
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from transformers import BertTokenizer, T5Tokenizer, LlamaTokenizer

from open_biomed.feature.base_featurizer import BaseFeaturizer
from open_biomed.feature.kg_featurizer import SUPPORTED_KG_FEATURIZER
from open_biomed.feature.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from open_biomed.utils.mol_utils import get_unimol_dictionary
from open_biomed.utils import to_clu_sparse, SmilesTokenizer, get_biot5_tokenizer

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

class MolOneHotFeaturizer(BaseFeaturizer):
    smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

    def __init__(self, config):
        super(MolOneHotFeaturizer, self).__init__()
        self.max_len = config["max_len"]
        self.enc = OneHotEncoder().fit(np.array(self.smiles_char).reshape(-1, 1))

    def __call__(self, data):
        temp = [c if c in self.smiles_char else '?' for c in data]
        if len(temp) < self.max_len:
            temp = temp + ['?'] * (self.max_len - len(temp))
        else:
            temp = temp[:self.max_len]
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)

class MolTransformerTokFeaturizer(BaseFeaturizer):
    name2tokenizer = {
        "bert": BertTokenizer,
        "t5": T5Tokenizer,
        "unimap": SmilesTokenizer,
        "3d-molm": LlamaTokenizer,
    }

    def __init__(self, config):
        super(MolTransformerTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.tokenizer = self.name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)
        if config["transformer_type"] == "3d-molm":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_special_tokens({'bos_token': '</s>'})
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
            self.tokenizer.add_special_tokens({'unk_token': '</s>'})
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
        if "prompt" in config:
            prompt = config["prompt"].split("<moleculeHere>")
            self.prompt = prompt[0] + "{content}"
            if len(prompt) > 1:
                self.prompt += prompt[1]
            self.len_prompt = len(self.tokenizer(self.prompt.format(content=""), max_length=self.max_length, padding=True, truncation=True).input_ids)
        else:
            self.prompt = "{content}"
            self.len_prompt = 0

    def __call__(self, data):
        result = self.tokenizer(self.prompt.format(content=data), max_length=self.max_length, padding=True, truncation=True)
        if len(result.input_ids) >= self.max_length:
            smi = self.tokenizer(data, max_length=self.max_length, truncation=True, add_special_tokens=False).input_ids[:self.max_length - self.len_prompt]
            result = self.tokenizer(self.prompt.format(content=self.tokenizer.decode(smi)), max_length=self.max_length, padding=True, truncation=True)
        return result

class MolSELFIESFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolSELFIESFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.tokenizer = get_biot5_tokenizer(config)
        self.add_special_tokens = False if "no_special_tokens" in config else True
        if "prompt" in config:
            prompt = config["prompt"].split("<moleculeHere>")
            self.prompt = prompt[0] + "{content}" + prompt[1]
        else:
            self.prompt = "{content}"

    def __call__(self, data):
        import selfies as sf
        mol = Chem.MolFromSmiles(data)
        smi = Chem.MolToSmiles(mol)
        sf_seq = '<bom>' + sf.encoder(smi, strict=False) + '<eom>'
        return self.tokenizer(self.prompt.format(content=sf_seq), max_length=self.max_length, padding=True, truncation=True, add_special_tokens=self.add_special_tokens)

class MolBPEFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolBPEFeaturizer, self).__init__()

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

class MolFPFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolFPFeaturizer, self).__init__()
        self.config = config
        self.which = config["which"]
        self.fp_size = config["fp_size"]
        self.radius = config["radius"]

    def _mol2np(self, mol, which, fp_size, radius):
        is_dict = False
        if which == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, useFeatures=False, useChirality=True)
        elif which == 'rdk':
            fp = Chem.RDKFingerprint(mol, fpSize=fp_size, maxPath=6)
        elif which == 'rdkc':
            # https://greglandrum.github.io/rdkit-blog/similarity/reference/2021/05/26/similarity-threshold-observations1.html
            # -- maxPath 6 found to be better for retrieval in databases
            fp = AllChem.UnfoldedRDKFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()
            is_dict = True
        elif which == 'morganc':
            fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=True, useBondTypes=True, useFeatures=True,  useCounts=True).GetNonzeroElements()
            is_dict = True
        elif which == 'topologicaltorsion':
            fp = AllChem.GetTopologicalTorsionFingerprint(mol).GetNonzeroElements()
            is_dict = True
        elif which == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif which == 'erg':
            v = AllChem.GetErGFingerprint(mol)
            fp = {idx:v[idx] for idx in np.nonzero(v)[0]}
            is_dict = True
        elif which == 'atompair':
            fp = AllChem.GetAtomPairFingerprint(mol).GetNonzeroElements()
            is_dict = True
        elif which == 'pattern':
            fp = Chem.PatternFingerprint(mol, fpSize=fp_size)
        elif which == 'ecfp4':
            # roughly equivalent to ECFP4
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size, useFeatures=False, useChirality=True)
        elif which == 'layered':
            fp = AllChem.LayeredFingerprint(mol, fpSize=fp_size, maxPath=7)
        elif which == 'mhfp':
            #TODO check if one can avoid instantiating the MHFP encoder
            fp = MHFPEncoder().EncodeMol(mol, radius=radius, rings=True, isomeric=False, kekulize=False, min_radius=1)
            fp = {f:1 for f in fp}
            is_dict = True
        elif not (type(which)==str):
            fp = which(mol)

        if is_dict:
            nd = np.zeros(fp_size)
            for k in fp:
                nk = k % fp_size #remainder
                if nd[nk] != 0:
                    nd[nk] = nd[nk] + fp[k] #pooling colisions
                nd[nk] = fp[k]
            return nd #np.log(1+nd) # discussion with segler
        return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')

    def __call__(self, data):
        mol = Chem.MolFromSmiles(data)
        """ + for folding * for concat """
        cc_symb = '*'
        if ('+' in self.which) or (cc_symb in self.which):
            concat = False
            split_sym = '+'
            if cc_symb in self.which:
                concat=True
                split_sym = '*'

            np_fp = np.zeros(self.fp_size)

            remaining_fps = (self.which.count(split_sym)+1)
            fp_length_remain = self.fp_size

            for fp_type in self.which.split(split_sym):
                if concat:
                    fpp = self._mol2np(mol, fp_type, fp_length_remain//remaining_fps, self.radius)
                    np_fp[(self.fp_size-fp_length_remain):(self.fp_size-fp_length_remain+len(fpp))] += fpp
                    fp_length_remain -= len(fpp)
                    remaining_fps -=1
                else:
                    try:
                        fpp = self._mol2np(mol, fp_type, self.fp_size, self.radius)
                        np_fp[:len(fpp)] += fpp
                    except:
                        pass
                    #print(fp_type,end='')

            np_fp = np.log(1 + np_fp)
        else:
            np_fp = self._mol2np(mol, self.which, self.fp_size, self.radius)
        if self.config["return_type"] == 'pt':
            return torch.FloatTensor(np_fp)
        else:
            return np_fp

class MolTGSAFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolTGSAFeaturizer, self).__init__()
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
        num_bond_features = 4  # bond type, bond stereo, is_conjugated
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

class MolGraphFeaturizer(BaseFeaturizer):
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
        'possible_bond_type_list':        [
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
        'possible_is_conjugated_list':    [False, True]
    }

    def __init__(self, config):
        super(MolGraphFeaturizer, self).__init__()
        self.config = config
        if self.config["name"] == "unimap":
            self.allowable_features["possible_atomic_num_list"] = self.allowable_features["possible_atomic_num_list"][:-1] + ['[MASK]', 'misc']
            self.allowable_features["possible_bond_type_list"] = self.allowable_features["possible_bond_type_list"][:-1] + ['[MASK]', '[SELF]', 'misc']
            self.allowable_features["possible_bond_stereo_list"] = self.allowable_features["possible_bond_stereo_list"] + ['[MASK]']
            self.allowable_features["possible_hybridization_list"] = self.allowable_features["possible_hybridization_list"][:-2] + ['misc']

    def __call__(self, data):
        if isinstance(data, str):
            mol = Chem.MolFromSmiles(data)
            # mol = AllChem.MolFromSmiles(data)
        else:
            mol = data
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            if self.config["name"] in ["ogb", "unimap"]:
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
            num_bond_features = 3 if self.config["name"] in ["ogb", "unimap"] else 2
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if self.config["name"] in ["ogb", "unimap"]:
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

class MolGGNNFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolGGNNFeaturizer, self).__init__()
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

class MolMGNNFeaturizer(BaseFeaturizer):
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
        super(MolMGNNFeaturizer, self).__init__()
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

class MolMultiScaleFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolMultiScaleFeaturizer, self).__init__()
        self.scales = config["scales"]
        self.featurizers = {}
        for scale in config["scales"]:
            conf = config[scale]
            self.featurizers[scale] = SUPPORTED_SINGLE_SCALE_MOL_FEATURIZER[conf["name"]](conf)

    def __call__(self, data):
        feat = {}
        for scale in self.scales:
            feat[scale] = self.featurizers[scale](data)
        return feat


# same with graphmvp   
class MolGraphFeaturizerV2(BaseFeaturizer):
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
        super(MolGraphFeaturizerV2, self).__init__()
        self.config = config

    def __call__(self, data):
        # mol = Chem.MolFromSmiles(data)
        mol = AllChem.MolFromSmiles(data)
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
            num_bond_features = 3 if self.config["name"] in "ogb" else 2
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

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates

def smi2_3Dcoords(smi, cnt=1, max_attempts=1000):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    logger.info("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=max_attempts, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    logger.info("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            logger.info("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

class MolConformationFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolConformationFeaturizer, self).__init__(config)
        assert config["allow_cache"], "Caching is recommended to speed up data processing"
        self.config = config

    def __call__(self, data):
        if data not in self.cached_data:
            self.cached_data[data] = smi2_3Dcoords(data, self.config["num_conformations"])
        return self.cached_data[data]

class MolUniMolFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolUniMolFeaturizer, self).__init__(config)
        assert config["allow_cache"], "Caching is recommended to speed up data processing"
        self.config = config
        self.dictionary = get_unimol_dictionary(config["dictionary_path"])

    def __call__(self, data):
        if data not in self.cached_data:
            smi = data
            mol = Chem.MolFromSmiles(smi)
            mol = AllChem.AddHs(mol)
            if len(mol.GetAtoms()) > 400 or len(mol.GetRingInfo().AtomRings()) > 20:
                coordinate_list = [smi2_2Dcoords(smi)]
                logger.info("atom num > 400 or Ring num > 20, use 2D coords for: %s" % smi)
            else:
                coordinate_list = smi2_3Dcoords(smi)
            atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])  # after add H 
            if self.config["remove_hydrogen"]:
                mask_hydrogen = atoms != 'H'
                if sum(mask_hydrogen) > 0:
                    atoms = atoms[mask_hydrogen]
                    for i in range(len(coordinate_list)):
                        coordinate_list[i] = coordinate_list[i][mask_hydrogen]

            # cropping
            if len(atoms) > self.config["max_n_atoms"] - 2:
                index = np.random.choice(len(atoms), self.config["max_n_atoms"] - 2, replace=False)
                atoms = np.array(atoms)[index]
                for i in range(len(coordinate_list)):
                    coordinate_list[i] = coordinate_list[i][index]

            # tokenize atoms
            atoms = torch.from_numpy(np.concatenate([np.array([self.dictionary.bos()]), self.dictionary.vec_index(atoms), np.array([self.dictionary.eos()])])).long()
            for i in range(len(coordinate_list)):
                coordinate_list[i] -= coordinate_list[i].mean(axis=0)
                coordinate_list[i] = np.concatenate([np.zeros((1, 3)), coordinate_list[i], np.zeros((1, 3))], axis=0)

            self.cached_data[data] = {
                'atoms': atoms, 
                'coordinates': coordinate_list
            }
        
        return self.cached_data[data]

class MolMultiModalFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolMultiModalFeaturizer, self).__init__()
        self.modality = config["modality"]
        self.featurizers = {}
        if "structure" in config["modality"]:
            conf = config["featurizer"]["structure"]
            self.featurizers["structure"] = SUPPORTED_SINGLE_MODAL_MOL_FEATURIZER[conf["name"]](conf)
        if "kg" in config["modality"]:
            conf = config["featurizer"]["kg"]
            self.featurizers["kg"] = SUPPORTED_KG_FEATURIZER[conf["name"]](conf)
        if "text" in config["modality"]:
            conf = config["featurizer"]["text"]
            self.featurizers["text"] = SUPPORTED_TEXT_FEATURIZER[conf["name"]](conf)

    def set_mol2kgid_dict(self, mol2kgid):
        self.featurizers["kg"].set_transform(mol2kgid)

    def set_mol2text_dict(self, mol2text):
        self.featurizers["text"].set_transform(mol2text)
        self.featurizers["text"].transform_count = {}

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

class MolEnsembleFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(MolEnsembleFeaturizer, self).__init__()
        self.featurizers = {}
        for model in config["models"]:
            self.featurizers[model] = SUPPORTED_MOL_FEATURIZER[config[model]["name"]](config[model])
    
    def __call__(self, data):
        feat = {}
        for model in self.featurizers:
            feat[model] = self.featurizers[model](data)
        return feat

SUPPORTED_SINGLE_SCALE_MOL_FEATURIZER = {
    "OneHot": MolOneHotFeaturizer,
    "KV-PLM*": MolBPEFeaturizer,
    "transformer": MolTransformerTokFeaturizer,
    "selfies": MolSELFIESFeaturizer,
    "fingerprint": MolFPFeaturizer,
    "TGSA": MolTGSAFeaturizer,
    "ogb": MolGraphFeaturizer,
    "unimap": MolGraphFeaturizer,
    "MGNN": MolMGNNFeaturizer,
    "BaseGNN": MolGraphFeaturizer,
    "conformation": MolConformationFeaturizer,
    "unimol": MolUniMolFeaturizer,
    "Ensemble": MolEnsembleFeaturizer,
}

SUPPORTED_SINGLE_MODAL_MOL_FEATURIZER = copy.deepcopy(SUPPORTED_SINGLE_SCALE_MOL_FEATURIZER)
SUPPORTED_SINGLE_MODAL_MOL_FEATURIZER["MultiScale"] = MolMultiScaleFeaturizer

SUPPORTED_MOL_FEATURIZER = copy.deepcopy(SUPPORTED_SINGLE_MODAL_MOL_FEATURIZER)
SUPPORTED_MOL_FEATURIZER["MultiModal"] = MolMultiModalFeaturizer

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
    data = MolGraphFeaturizer({"name": "ogb"})(smi)
    print(data.x, data.edge_index, data.edge_attr)

    smi = "C(C(C(=O)O)N)C(C(=O)O)O"
    data = MolBPEFeaturizer({
        "name": "KV-PLM*",
        "code_name": "../assets/KV-PLM*/bpe_coding.txt",
        "vocabulary": "../assets/KV-PLM*/bpe_vocab.txt",
        "vocabulary_threshold": 80,
        "max_length": 32
    })(smi)
    print(data)

    smi = "OC(=O)C1=CC(=CC=C1O)\\N=N\\C1=CC=C(C=C1)S(=O)(=O)NC1=NC=CC=C1"
    data = MolMGNNFeaturizer({})(smi)
    print(data.x, data.edge_index)

def featurize_file(args):
    with open(args.smiles_file, "r") as f:
        smis = [line.rstrip("\n") for line in f.readlines()]
    config = json.load(open(args.config_file, "r"))
    featurizer = SUPPORTED_MOL_FEATURIZER[args.featurizer](config)
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