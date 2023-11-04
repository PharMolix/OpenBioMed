import os
import pickle
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)


def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeNetGraphDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

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


def graph_data_obj_to_nx_simple(data):
    """ torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        temp_feature = atom_features[i]
        G.add_node(
            i,
            x0=temp_feature[0],
            x1=temp_feature[1],
            x2=temp_feature[2],
            x3=temp_feature[3],
            x4=temp_feature[4],
            x5=temp_feature[5],
            x6=temp_feature[6],
            x7=temp_feature[7],
            x8=temp_feature[8])
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        temp_feature= edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       e0=temp_feature[0],
                       e1=temp_feature[1],
                       e2=temp_feature[2])

    return G


def nx_to_graph_data_obj_simple(G):
    """ vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered. """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['x0'], node['x1'], node['x2'], node['x3'], node['x4'], node['x5'], node['x6'], node['x7'], node['x8']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['e0'], edge['e1'], edge['e2']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def create_standardized_mol_id(smiles):
    """ smiles -> inchi """

    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)\
            # c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
    return


class MoleculeNetGraphDataset(InMemoryDataset):
    def __init__(self, root, dataset='zinc250k', transform=None,
                 pre_transform=None, pre_filter=None, empty=False):

        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(MoleculeNetGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        if self.dataset == 'davis':
            file_name_list = ['davis']
        elif self.dataset == 'kiba':
            file_name_list = ['kiba']
        else:
            file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):

        def shared_extractor(smiles_list, rdkit_mol_objs, labels):
            data_list, data_smiles_list, data_label_list = [], [], []
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=1)
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is None:
                    continue
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])
            return data_list, data_smiles_list, data_label_list

        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'malaria':
            smiles_list, rdkit_mol_objs, labels = \
                _load_malaria_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'cep':
            smiles_list, rdkit_mol_objs, labels = \
                _load_cep_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'pcba':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)
        
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            data_list, data_smiles_list, data_label_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data_label_array = np.array(data_label_list)
        saver_path = os.path.join(self.processed_dir, 'labels')
        np.savez_compressed(saver_path, labels=data_label_array)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return


def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values


def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_freesolv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_malaria_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['activity']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_cep_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['PCE']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_sider_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively """

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one """

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]
