import pickle
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from utils import ToDevice
from utils.mol_utils import load_mol2vec

class Text2MolMLP(nn.Module):
    def __init__(self, ninp, nout, nhid, model_name_or_path, cid2smiles_path, cid2vec_path, mol2vec_output_path=None):
        super(Text2MolMLP, self).__init__()
        if mol2vec_output_path is not None:
            self.smiles2vec = load_mol2vec(mol2vec_output_path)
        else:
            self._prepare_smi2vec(cid2smiles_path, cid2vec_path)
        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout

        self.mol_hidden1 = nn.Linear(nout, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)
        
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter('temp', self.temp)

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = BertModel.from_pretrained(model_name_or_path)
        self.text_tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    def _prepare_smi2vec(self, cid2smiles_path, cid2vec_path):
        cid2smiles = pickle.load(open(cid2smiles_path, "rb"))
        cid2remove = []
        for cid in cid2smiles:
            if cid2smiles[cid] == '*':
                cid2remove.append(cid)
        for cid in cid2remove:
            cid2smiles.pop(cid, None)

        smiles2cid = {}
        for cid in cid2smiles:
            smi = cid2smiles[cid]
            smi = smi.replace("\\\\", "\\")
            if cid2smiles[cid] == '*':
                continue
            smiles2cid[smi] = cid

        cid2vec = {}
        with open(cid2vec_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
            for line in reader:
                cid2vec[line['cid']] = np.fromstring(line['mol2vec'], sep=" ")
        self.smiles2vec = {}
        for smi in smiles2cid:
            if smiles2cid[smi] in cid2vec:
                self.smiles2vec[smi] = cid2vec[smiles2cid[smi]]

    def forward(self, smi, text, device):
        text = self.text_tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        text = ToDevice(text, device)
        text_encoder_output = self.text_transformer_model(**text)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)

        smi = smi.replace("\\\\", "\\")
        mol_x = torch.from_numpy(self.smiles2vec[smi]).reshape((1, -1)).to(device).float()
        x = self.relu(self.mol_hidden1(mol_x))
        x = self.relu(self.mol_hidden2(x))
        x = self.mol_hidden3(x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return F.cosine_similarity(x, text_x)