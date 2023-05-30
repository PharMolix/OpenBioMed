import argparse
import csv
import numpy as np
import pickle

import rdkit.Chem as Chem
from rdkit.Chem import MolStandardize
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import torch

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        if mol is not None:
            return True
        else:
            return False
    except:
        return False

def can_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        standardizer = MolStandardize.normalize

        # standardize the molecule
        standardized_mol = standardizer.Normalizer().normalize(mol)
        # get the standardized SMILES string
        standardized_smiles = Chem.MolToSmiles(standardized_mol, isomericSmiles=True)
    except:
        standardized_smiles = smi

    return standardized_smiles

def write_sdf(cid2smiles_file, output_file, sdf_file):
    cid2smiles = pickle.load(open(cid2smiles_file, "rb"))
    smiles2cid = {}
    for cid in cid2smiles:
        if cid2smiles[cid] != '*':
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(cid2smiles[cid]))
            smiles2cid[smi] = cid
    all_mols = []

    print("Loading output file...")
    with open(output_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            line = line.rstrip("\n").split("\t")
            try:
                gt_smi = Chem.MolToSmiles(Chem.MolFromSmiles(line[1]))
                output_mol = Chem.MolFromSmiles(line[2])
                if output_mol is not None:
                    output_mol.SetProp("CID", smiles2cid[gt_smi])
                    all_mols.append(output_mol)
            except:
                continue

    print("Writing sdf file...")
    with Chem.SDWriter(sdf_file) as f:
        for mol in all_mols:
            f.write(mol)

def load_mol2vec(file):
    mol2vec = {}
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        for row in reader:
            mol_str = " ".join(row[-300:])
            mol2vec[row[3]] = np.fromstring(mol_str, sep=" ")
    return mol2vec

def link_datasets(source, target):
    targetsmi2id = {}
    for i, smi in enumerate(target.smiles):
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            targetsmi2id[smi] = i
        except:
            continue
    match_indexes = []
    for smi in source.smiles:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            match_indexes.append(targetsmi2id[smi])
        except:
            match_indexes.append(-1)
    return match_indexes

def convert_pyg_batch(output, batch_idx, max_n_nodes):
    batch_size = torch.max(batch_idx).item() + 1
    batch_output = []
    batch_attention_mask = []
    for i in range(batch_size):
        feat = output[torch.where(batch_idx == i)]
        if feat.shape[0] < max_n_nodes:
            batch_output.append(torch.cat((
                feat,
                torch.zeros(max_n_nodes - feat.shape[0], feat.shape[1]).to(feat.device)
            ), dim=0))
            batch_attention_mask.append(torch.cat((
                torch.ones(feat.shape[0]).to(feat.device), 
                torch.zeros(max_n_nodes - feat.shape[0]).to(feat.device)
            ), dim=0))
        else:
            batch_output.append(feat[:max_n_nodes, :])
            batch_attention_mask.append(torch.ones(max_n_nodes).to(feat.device))
    batch_output = torch.stack(batch_output, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    return batch_output, batch_attention_mask
  

def convert_kge_batch(all_neigh_feats, max_n_nodes=16):
    batch_size = len(all_neigh_feats)
    batch_output = []
    batch_attention_mask = []
    for i in range(batch_size):
        feats = all_neigh_feats[i]
        if feats.size(0) < max_n_nodes:
            batch_output.append(torch.cat((
                feats,
                torch.zeros(max_n_nodes - feats.shape[0], feats.shape[1]).to(feats.device)
            ), dim=0))
            batch_attention_mask.append(torch.cat((
                torch.ones(feats.shape[0]).to(feats.device), 
                torch.zeros(max_n_nodes - feats.shape[0]).to(feats.device)
            ), dim=0))
        else:
            batch_output.append(feats[:max_n_nodes, :])
            batch_attention_mask.append(torch.ones(max_n_nodes).to(feats.device))
    batch_output = torch.stack(batch_output, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    return batch_output, batch_attention_mask
    #neigh_attn = torch.ones((neigh_feats.size(0), 1)).to(node_attention_mask.device).type(node_attention_mask.type())
    #neigh_feats = self.kg_linear(neigh_feats).unsqueeze(1)
    #neigh_feats = self.kg_linear(neigh_feats).unsqueeze(1)


def add_argument(parser):
    parser.add_argument("--mode", type=str, choices=["write_sdf", "unittest"])

def add_sdf_argument(parser):
    parser.add_argument("--cid2smiles_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--sdf_file", type=str, default="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args, _ = parser.parse_known_args()

    if args.mode == "write_sdf":
        add_sdf_argument(parser)
        args = parser.parse_args()
        write_sdf(args.cid2smiles_file, args.output_file, args.sdf_file)