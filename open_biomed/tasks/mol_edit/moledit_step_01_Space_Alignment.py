import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tqdm import tqdm
import time
import json
import re
import copy
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader


from utils.molstm_utils import get_molecule_repr_MoleculeSTM
from models.multimodal.moleculestm import MLP
from utils.molstm_utils import load_molecule_models
from utils.molstm_utils import freeze_network
from datasets.moledit_dataset import SUPPORTED_MOLEDIT_DATASET
from models.task_model.moledit_model import MoleditModel
from models.multimodal.mega_molbart.mega_mol_bart import MegaMolBART

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.SSL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        SSL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        SSL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        SSL_acc = SSL_acc.detach().cpu().item()
        
    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        SSL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        SSL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    
    elif args.SSL_loss == 'RR':
        criterion = nn.MSELoss()
        SSL_loss = criterion(X, Y)
        SSL_acc = 0

    else:
        raise Exception

    return SSL_loss, SSL_acc

def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def get_molecule_repr_generation(molecule_data, molecule_model, molecule_type="MegaMolBART", MegaMolBART_wrapper=None):
    if molecule_type == "MegaMolBART":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding_model_given(molecule_model, molecule_data)  # [pad, B, d], [pad, B]
        molecule_repr = mean_pooling(embedding, pad_mask)
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    return molecule_repr


def save_model(save_best, epoch=None):
    if args.output_path is not None:
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            model_file = "model.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        saved_file_path = os.path.join(args.output_path, "generation2MoleculeSTM_{}".format(model_file))
        torch.save(generation2MoleculeSTM.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_path, "MoleculeSTM2generation_{}".format(model_file))
        torch.save(MoleculeSTM2generation.state_dict(), saved_file_path)
    return


def train(epoch):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    batch_num=0

    for batch in L:
        if args.MoleculeSTM_molecule_type == "SMILES":
            SMILES_list = batch["structure"]["SMILES"]
        else:
            SMILES_list = batch["structure"]["SMILES"]
            graph = batch["structure"]["graph"]
            graph = graph.to(device)

        if args.use_static_files==1:
            molecule_repr_MoleculeSTM = molecule_repr_MoleculeSTM_list[batch_num].to(device)
            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)
            molecule_repr_generation = molecule_repr_generation_list[batch_num].to(device)
            molecule_repr_generation2MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)
            batch_num+=1
        else:
            if args.MoleculeSTM_molecule_type == "SMILES":
                molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
                    SMILES_list, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                    molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper
                )
            else:
                molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
                    graph, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                    molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper
                )
            if args.generation_model == "MegaMolBART":
                molecule_repr_generation = get_molecule_repr_generation(
                    SMILES_list, molecule_model=molecule_model_generation,
                    molecule_type="MegaMolBART", MegaMolBART_wrapper=MegaMolBART_wrapper
                )
            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)
            molecule_repr_generation2MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)

        loss_01, acc_01 = do_CL(molecule_repr_generation, molecule_repr_MoleculeSTM2generation, args)
        loss_02, acc_02 = do_CL(molecule_repr_MoleculeSTM, molecule_repr_generation2MoleculeSTM, args)
        loss = (loss_01 + loss_02) / 2
        acc = (acc_01 + acc_02) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_acc += acc

    accum_loss /= len(L)
    accum_acc /= len(L)
    
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True, epoch=epoch)
    print("SSL Loss: {:.5f}\tSSL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return

def generate_static_files():
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    molecule_repr_MoleculeSTM_list = [] 
    molecule_repr_generation_list = []
    for batch in L:
        if args.MoleculeSTM_molecule_type == "SMILES":
            SMILES_list = batch["structure"]["SMILES"]
        else:
            SMILES_list = batch["structure"]["SMILES"]
            graph = batch["structure"]["graph"]
            graph = graph.to(device)
        if args.MoleculeSTM_molecule_type == "SMILES":
            molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
                SMILES_list, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper
            )
        else:
            molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
                graph, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper
            )
            molecule_repr_MoleculeSTM_list.append(molecule_repr_MoleculeSTM)   
        if args.generation_model == "MegaMolBART":
            molecule_repr_generation = get_molecule_repr_generation(
                SMILES_list, molecule_model=molecule_model_generation,
                molecule_type="MegaMolBART", MegaMolBART_wrapper=MegaMolBART_wrapper
            )
        molecule_repr_generation_list.append(molecule_repr_generation)   
    saved_file_path = os.path.join(args.static_files_path, "molecule_repr_MoleculeSTM_list.pkl")
    with open(saved_file_path, 'wb') as f:  
        pickle.dump(molecule_repr_MoleculeSTM_list, f) 
    saved_file_path = os.path.join(args.static_files_path, "molecule_repr_generation_list.pkl")
    with open(saved_file_path, 'wb') as f:  
        pickle.dump(molecule_repr_generation_list, f) 
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="./datasets/mol_edit/ZINC250K_data")
    parser.add_argument("--static_files_path", type=str, default="./datasets/mol_edit/ZINC250K_data/static_files/molkformer-Graph")
    parser.add_argument("--dataset", type=str, default="ZINC250K")
    parser.add_argument("--MoleculeSTM_molecule_type", type=str, default="Graph", choices=["SMILES", "Graph"])
    parser.add_argument("--output_path", type=str, default="./ckpts/finetune_ckpts/moledit/molkformer/Graph")
    parser.add_argument("--config_path", type=str, default="./configs/moledit/molkformer-Graph-MegaMolBART.json")
    parser.add_argument("--mode", type=str, default="train")
    ########## for MoleculeSTM ##########
    parser.add_argument("--MoleculeSTM_model_dir", type=str, default=None)
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for generation ##########
    parser.add_argument('--generation_model', type=str, default="MegaMolBART", choices=["MegaMolBART"])

    ######### for MegaMolBART ##########
    parser.add_argument("--MegaMolBART_generation_model_dir", type=str, default="./ckpts/fusion_ckpts/pretrained_MegaMolBART/checkpoints")
    parser.add_argument("--vocab_path", type=str, default="./ckpts/fusion_ckpts/pretrained_MegaMolBART/bart_vocab.txt")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--generation_lr", type=float, default=1e-2)
    parser.add_argument("--MoleculeSTM_lr", type=float, default=1e-2)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE", "RR"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument('--use_normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    parser.add_argument("--MASTER_PORT", type=str, default='6000')
    parser.add_argument("--use_processed_dataset_250K", type=int, default=0)
    parser.add_argument("--generate_static_files", type=int, default=0)
    parser.add_argument("--use_static_files", type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    
    config = json.load(open(args.config_path)) 
    os.environ['MASTER_PORT'] = args.MASTER_PORT

    # load dataset
    if args.use_processed_dataset_250K==1: # skip SUPPORTED_MOLEDIT_DATASET
        with open("./datasets/mol_edit/ZINC250K_data/dataset_zinc250K.pkl", "rb") as f:  
             dataset = pickle.load(f)
    else:     
        dataset = SUPPORTED_MOLEDIT_DATASET[args.dataset](args.dataset_path, config["data"]["mol"], split="train")

    dataloader_class = pyg_DataLoader

    device = torch.device(args.device) \
        if torch.cuda.is_available() else torch.device("cpu")
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # load model
    molecule_model_MoleculeSTM = MoleditModel(config["network"])
    mol2latent_MoleculeSTM = None
    if config["model"]== "molstm-MegaMolBART":
        MegaMolBART_wrapper = molecule_model_MoleculeSTM.model.MegaMolBART_wrapper
        molecule_model_generation = copy.deepcopy(MegaMolBART_wrapper.model)
    else: 
        MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.MegaMolBART_generation_model_dir, output_dir=None)
        molecule_model_generation = copy.deepcopy(MegaMolBART_wrapper.model)

    torch.cuda.set_device(int(re.search(r'\d+', args.device).group()))

    molecule_model_generation = molecule_model_generation.to(device)
    molecule_model_MoleculeSTM = molecule_model_MoleculeSTM.to(device)
    freeze_network(molecule_model_generation)
    freeze_network(molecule_model_MoleculeSTM)
    molecule_model_generation.eval()
    molecule_model_MoleculeSTM.eval()


    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    molecule_dim_generation = 256
    molecule_dim_MoleculeSTM = args.SSL_emb_dim
    generation2MoleculeSTM = MLP(molecule_dim_generation, [molecule_dim_MoleculeSTM, molecule_dim_MoleculeSTM]).to(device)
    MoleculeSTM2generation = MLP(molecule_dim_MoleculeSTM, [molecule_dim_generation, molecule_dim_generation]).to(device)

    model_param_group = [
        {"params": generation2MoleculeSTM.parameters(), "lr": args.generation_lr},
        {"params": MoleculeSTM2generation.parameters(), "lr": args.MoleculeSTM_lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    if args.generate_static_files==1: 
        generate_static_files()

    if args.use_static_files==1:
        saved_file_path = os.path.join(args.static_files_path, "molecule_repr_MoleculeSTM_list.pkl")
        with open(saved_file_path, 'rb') as f:  
            molecule_repr_MoleculeSTM_list = pickle.load(f) 
        saved_file_path = os.path.join(args.static_files_path, "molecule_repr_generation_list.pkl")
        with open(saved_file_path, 'rb') as f:  
            molecule_repr_generation_list = pickle.load(f) 
        

    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        train(e)
