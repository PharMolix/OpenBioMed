import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from itertools import repeat

import torch
from torch_geometric.data import Data, InMemoryDataset

from models.MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple


class ZINC250K_Dataset_Graph(InMemoryDataset):
    def __init__(self, root, subset_size=257, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.SMILES_file = os.path.join(self.root, "raw/250k_rndm_zinc_drugs_clean_3.csv")
        df = pd.read_csv(self.SMILES_file)
        SMILES_list = df['smiles'].tolist()
        self.SMILES_list = [x.strip() for x in SMILES_list]

        super(ZINC250K_Dataset_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_molecule_only')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            RDKit_mol = Chem.MolFromSmiles(SMILES)
            graph = mol_to_graph_data_obj_simple(RDKit_mol)
            graph_list.append(graph)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return SMILES, data

    def __len__(self):
        return len(self.SMILES_list)