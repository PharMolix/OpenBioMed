from torch.utils.data import Dataset
import os
import pandas as pd


class ZINC250K_Dataset_SMILES(Dataset):
    def __init__(self, root, subset_size=512):
        self.root = root

        SMILES_file = os.path.join(self.root, "raw/250k_rndm_zinc_drugs_clean_3.csv")
        df = pd.read_csv(SMILES_file)
        SMILES_list = df['smiles'].tolist()  # Already canonical SMILES
        self.SMILES_list = [x.strip() for x in SMILES_list]
        # self.SMILES_list = [{'original_tokens': d, 'masked_pad_masks': [1,2,3]} for d in self.SMILES_list]
       
        new_SMILES_file = os.path.join(self.root, "raw/smiles.csv")
        if not os.path.exists(new_SMILES_file):
            data_smiles_series = pd.Series(self.SMILES_list)
            print("saving to {}".format(new_SMILES_file))
            data_smiles_series.to_csv(new_SMILES_file, index=False, header=False)

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return
    
    def __getitem__(self, index):
        SMILES = self.SMILES_list[index]
        return SMILES

    def __len__(self):
        return len(self.SMILES_list)
