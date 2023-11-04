import os
from itertools import repeat
import pandas as pd
import json
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from models.MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple

from models.MoleculeSTM.datasets import PubChemSTM_Datasets_SMILES


class PubChemSTM_Datasets_Raw_SMILES(PubChemSTM_Datasets_SMILES):
    def __init__(self, root):
        self.root = root

        CID2text_file = os.path.join(self.root, "raw/CID2text_raw.json")
        # Both PubChemSTM and PubChemSTM_Raw share the same CID2SMILES file.
        CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
        self.load_CID2SMILES(CID2text_file, CID2SMILES_file)
        
        self.text_list = []
        missing_count = 0
        for CID, value_list in self.CID2text_data.items():
            if CID not in self.CID2SMILES:
                print("CID {} missing".format(CID))
                missing_count += 1
                continue
            for value in value_list:
                self.text_list.append([CID, value])
        print("missing", missing_count)
        print("len of text_list: {}".format(len(self.text_list)))

        return


class PubChemSTM_SubDatasets_Raw_SMILES(PubChemSTM_Datasets_Raw_SMILES):
    def __init__(self, root, size):
        self.root = root

        CID2text_file = os.path.join(self.root, "raw/CID2text_raw.json")
        CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
        self.load_CID2SMILES(CID2text_file, CID2SMILES_file)
        
        self.text_list = []
        for CID, value_list in self.CID2text_data.items():
            if CID not in self.CID2SMILES:
                print("CID {} missing".format(CID))
                continue
            for value in value_list:
                self.text_list.append([CID, value])
            if len(self.text_list) >= size:
                break
        print("len of text_list: {}".format(len(self.text_list)))
        return


class PubChemSTM_Datasets_Raw_Graph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text_raw.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed_raw/CID_text_list.csv")
        
        super(PubChemSTM_Datasets_Raw_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_raw')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        suppl = Chem.SDMolSupplier(self.SDF_file_path)

        CID2graph = {}
        for mol in tqdm(suppl):
            CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)
            CID2graph[CID] = graph
        print("CID2graph", len(CID2graph))
        
        with open(self.CID2text_file, "r") as f:
            CID2text_data = json.load(f)
        print("CID2data", len(CID2text_data))
            
        CID_list, graph_list, text_list = [], [], []
        for CID, value_list in CID2text_data.items():
            CID = int(CID)
            if CID not in CID2graph:
                print("CID {} missing".format(CID))
                continue
            graph = CID2graph[CID]
            for value in value_list:
                text_list.append(value)
                CID_list.append(CID)
                graph_list.append(graph)

        CID_text_df = pd.DataFrame({"CID": CID_list, "text": text_list})
        CID_text_df.to_csv(self.CID_text_file_path, index=None)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return
    
    def load_Graph_CID_and_text(self):
        self.graphs, self.slices = torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.CID_text_file_path)
        self.CID_list = CID_text_df["CID"].tolist()
        self.text_list = CID_text_df["text"].tolist()
        return

    def get(self, idx):
        text = self.text_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return text, data

    def __len__(self):
        return len(self.text_list)


class PubChemSTM_SubDatasets_Raw_Graph(PubChemSTM_Datasets_Raw_Graph):
    def __init__(self, root, size, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.size = size
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.size = size
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text_raw.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed_raw/CID_text_list.csv")
        
        super(PubChemSTM_SubDatasets_Raw_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    def __len__(self):
        return self.size
