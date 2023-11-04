import os
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset


class MoleculeNetSMILESDataset(Dataset):
    def __init__(self, root):
        '''
        This needs to be called after calling the MoleculeNetGraphDataset.
        '''
        self.root = root
        SMILES_file = os.path.join(root, "processed", "smiles.csv")

        self.SMILES_list = []
        with open(SMILES_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                SMILES = line.strip()
                mol = Chem.MolFromSmiles(SMILES)
                canon_SMILES = Chem.MolToSmiles(mol)
                self.SMILES_list.append(canon_SMILES)

        labels_file = os.path.join(root, "processed", "labels.npz")
        self.labels_data = np.load(labels_file)['labels']

        print(len(self.SMILES_list), '\t', self.labels_data.shape)
        return

    def __getitem__(self, index):
        SMILES = self.SMILES_list[index]
        labels = self.labels_data[index]
        return SMILES, labels

    def __len__(self):
        return len(self.SMILES_list)
