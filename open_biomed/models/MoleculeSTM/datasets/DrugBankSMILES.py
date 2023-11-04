import os
from torch.utils.data import Dataset


class DrugBank_Datasets_SMILES_retrieval(Dataset):
    def __init__(self, root, train_mode, neg_sample_size, template="SMILES_description_{}.txt"):
        self.root = root

        self.SMILES_list, self.text_list = [], []
        SMILES2description_file = os.path.join(self.root, "raw", template.format(train_mode))
        f = open(SMILES2description_file, 'r')
        for line in f.readlines():
            line = line.strip().split("\t", 1)
            SMILES = line[0]
            text = line[1]
            self.SMILES_list.append(SMILES)
            self.text_list.append(text)

        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", template.format(train_mode))
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list
        return

    def __getitem__(self, index):
        description = self.text_list[index]
        SMILES = self.SMILES_list[index]
        
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_description = [self.text_list[idx] for idx in neg_index_list]
        
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_SMILES = [self.SMILES_list[idx] for idx in neg_index_list]

        return description, SMILES, neg_description, neg_SMILES

    def __len__(self):
        return len(self.SMILES_list)


class DrugBank_Datasets_SMILES_ATC(Dataset):
    def __init__(self, root, file_name, neg_sample_size, prompt_template="{}."):
        self.root = root
        self.neg_sample_size = neg_sample_size
        self.prompt_template = prompt_template
    
        SMILES2ATC_txt_file = os.path.join(self.root, 'raw', file_name)
        
        f = open(SMILES2ATC_txt_file, 'r')
        SMILES_list, ATC_code_list, ATC_label_list = [], [], []
        for line in f.readlines():
            line = line.strip().split("\t")
            SMILES_list.append(line[0])
            ATC_code_list.append(line[1])
            ATC_label_list.append(prompt_template.format(line[2]))

        self.SMILES_list = SMILES_list
        self.ATC_code_list = ATC_code_list
        self.ATC_label_list = ATC_label_list

        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", file_name)
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        assert len(self.SMILES_list) == len(self.neg_index_list) == len(ATC_code_list) == len(ATC_label_list)
        return

    def __getitem__(self, index):
        text = self.ATC_label_list[index]
        SMILES = self.SMILES_list[index]
        
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.ATC_label_list[idx] for idx in neg_index_list]
        
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_SMILES = [self.SMILES_list[idx] for idx in neg_index_list]

        return text, SMILES, neg_text, neg_SMILES

    def __len__(self):
        return len(self.SMILES_list)