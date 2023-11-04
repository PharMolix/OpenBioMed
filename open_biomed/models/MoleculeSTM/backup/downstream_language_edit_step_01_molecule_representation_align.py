import argparse
import os
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader

from MoleculeSTM.utils import get_molecule_repr_MoleculeSTM
from MoleculeSTM.downstream_language_edit_utils import load_molecule_models
from MoleculeSTM.utils import freeze_network
from MoleculeSTM.datasets import PubChem_Datasets_Only_SMILES


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
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def get_molecule_repr_generation(molecule_data, molecule_model, molecule_type="MegaMolBART", MegaMolBART_wrapper=None):
    if molecule_type == "MegaMolBART":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding_model_given(molecule_model, molecule_data)  # [pad, B, d], [pad, B]
        # molecule_repr = embedding[0, :, :]  # [B, d]
        # next we will take the mean pooling instead of the CLS token.
        molecule_repr = mean_pooling(embedding, pad_mask)
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    return molecule_repr


def save_model(save_best, epoch=None):
    if args.output_model_dir is not None:
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            model_file = "model.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        saved_file_path = os.path.join(args.output_model_dir, "generation2MoleculeSTM_{}".format(model_file))
        torch.save(generation2MoleculeSTM.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "MoleculeSTM2generation_{}".format(model_file))
        torch.save(MoleculeSTM2generation.state_dict(), saved_file_path)
    return


def train(epoch):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    for batch in L:
        SMILES_list = batch

        molecule_repr_generation = get_molecule_repr_generation(
            SMILES_list, molecule_model=molecule_model_generation,
            molecule_type="MegaMolBART", MegaMolBART_wrapper=MegaMolBART_wrapper
        )
        molecule_repr_generation2MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)

        molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
            SMILES_list, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
            molecule_type="MegaMolBART", MegaMolBART_wrapper=MegaMolBART_wrapper
        )
        molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)

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
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}Time: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--dataspace_path", type=str, default="../../Datasets")
    parser.add_argument("--dataset", type=str, default="PubChem")
    parser.add_argument("--molecule_type", type=str, default="MegaMolBART", choices=["MegaMolBART", "Graph"])
    parser.add_argument("--output_model_dir", type=str, default=None)

    ########## for MoleculeSTM ##########
    parser.add_argument("--MoleculeSTM_model_dir", type=str, default="../../pretrained_model_Raw")
    parser.add_argument("--SSL_emb_dim", type=int, default=256)

    ########## for generation ##########
    parser.add_argument("--generation_model_dir", type=str, default="../../Datasets/pretrained_MegaMolBART/checkpoints")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--generation_lr", type=float, default=1e-4)
    parser.add_argument("--MoleculeSTM_lr", type=float, default=1e-4)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    print(args)

    MegaMolBART_wrapper, molecule_model_generation, molecule_dim_generation, \
        molecule_model_MoleculeSTM, mol2latent_MoleculeSTM, molecule_dim_MoleculeSTM = load_molecule_models(args)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    molecule_model_generation = molecule_model_generation.to(device)
    molecule_model_MoleculeSTM = molecule_model_MoleculeSTM.to(device)
    mol2latent_MoleculeSTM = mol2latent_MoleculeSTM.to(device)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    freeze_network(molecule_model_generation)
    freeze_network(mol2latent_MoleculeSTM)
    freeze_network(molecule_model_MoleculeSTM)
    molecule_model_generation.eval()
    mol2latent_MoleculeSTM.eval()
    molecule_model_MoleculeSTM.eval()

    if args.molecule_type == "MegaMolBART":
        if "PubChem" in args.dataset:
            dataset_root = os.path.join(args.dataspace_path, "PubChem_data")
        else:
            raise Exception
        dataset = PubChem_Datasets_Only_SMILES(dataset_root)
        dataloader_class = torch_DataLoader
    else:
        raise Exception
    
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    generation2MoleculeSTM = nn.Linear(molecule_dim_generation, molecule_dim_MoleculeSTM).to(device)
    MoleculeSTM2generation = nn.Linear(molecule_dim_MoleculeSTM, molecule_dim_generation).to(device)

    model_param_group = [
        {"params": generation2MoleculeSTM.parameters(), "lr": args.generation_lr},
        {"params": MoleculeSTM2generation.parameters(), "lr": args.MoleculeSTM_lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10
    
    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        train(e)
