import argparse
import math
import numpy as np
from rdkit import Chem, RDLogger

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from downstream_language_edit_utils import load_language_molecule_and_edit_models, clip_loss_for_edit, evaluate_SMILES_list
from MoleculeSTM.utils import prepare_text_tokens


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)

    ########## for editing ##########
    parser.add_argument("--description", type=str)
    parser.add_argument("--input_model_dir", type=str)
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"])
    parser.add_argument("--input_SMILES", type=str, default=None)
    parser.add_argument("--l2_lambda", type=float, default=0.008)

    ########## for ? ##########
    parser.add_argument("--dataspace_path", type=str, default="../../Datasets")
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)

    ########## for MoleculeSTM ##########
    parser.add_argument("--MoleculeSTM_model_dir", type=str, default="../../pretrained_model_Raw")

    ########## for generation ##########
    parser.add_argument("--generation_model_dir", type=str, default="../../Datasets/pretrained_MegaMolBART/checkpoints")

    ########## for MoleculeSTM and generation projection ##########
    parser.add_argument("--language_edit_model_dir", type=str, default="edit_temp/EBM_NCE")   

    ########## for editing ##########
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    print(args)

    text_model, text_tokenizer, text_dim, molecule_model, MegaMolBART_wrapper, molecule_dim, \
        text2latent, mol2latent, generation2MoleculeSTM, MoleculeSTM2generation = load_language_molecule_and_edit_models(args)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    text_model = text_model.to(device)
    molecule_model = molecule_model.to(device)
    text2latent = text2latent.to(device)
    mol2latent = mol2latent.to(device)
    generation2MoleculeSTM.to(device)
    MoleculeSTM2generation.to(device)
    text_model.eval()
    molecule_model.eval()
    text2latent.eval()
    mol2latent.eval()
    generation2MoleculeSTM.eval()
    MoleculeSTM2generation.eval()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    description_list = [args.description]
    text_tokens_ids, text_masks = prepare_text_tokens(
        device=device, description=description_list, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len)
    text_output = text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
    text_repr = text_output["pooler_output"]
    text_repr = text2latent(text_repr)

    record_SMILES_list = []

    if args.mode == "edit":
        SMILES_list = [args.input_SMILES]
        latent_code_init, pad_mask_init = MegaMolBART_wrapper.smileslist2embedding(SMILES_list)  # [pad, B, d], [pad, B]
        molecule_repr_generation_init = mean_pooling(latent_code_init, pad_mask_init) # [B, d]
        # record_SMILES_list.append(args.input_SMILES)
    else:
        padding_dim = 10
        latent_code_init = torch.randn(padding_dim, 1, molecule_dim).to(device)
        pad_mask_init = torch.zeros(padding_dim, 1).bool().to(device)
        print("latent_code_init", latent_code_init.size())
        print("pad_mask_init", pad_mask_init.size())

    generated_mols = MegaMolBART_wrapper.inverse_transform(
        [latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)
    print("initial SMILES", generated_mols[0])
    record_SMILES_list.append(generated_mols[0])

    l2_lambda_list = [
        1, 0.1, 0.01, 0.001, 0.0001,
        3, 0.3, 0.03, 0.003, 0.0003,
        5, 0.5, 0.05, 0.005, 0.0005,
        8, 0.8, 0.08, 0.008, 0.0008,
    ]
    l2_lambda_list = [
        0.1,
    ]

    for l2_lambda in l2_lambda_list:
        result_SMILES_list = [record_SMILES_list[0]]
        print("with lambda {} ......".format(l2_lambda))
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True
        optimizer = optim.Adam([latent], lr=args.lr)
        
        if args.verbose:
            L = tqdm(range(args.epochs))
        else:
            L = range(args.epochs)
        for i in L:
            t = i / args.epochs
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            molecule_repr_generation = mean_pooling(latent, pad_mask_init) # [B, d]
            # molecule_repr_MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)

            clip_loss_ = clip_loss_for_edit(molecule_repr_generation, mol2latent, text_repr)
            l2_loss_ =  args.l2_lambda * ((latent_code_init - latent) ** 2).sum()

            loss = clip_loss_ + l2_loss_
            print(clip_loss_.item(), l2_loss_.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print("clip loss: {:.5f}\tL2 loss: {:.5f}".format(clip_loss_.item(), args.l2_lambda * l2_loss_))

        generated_mols = MegaMolBART_wrapper.inverse_transform(
            [latent], pad_mask_init.bool().cuda(), k=1, sanitize=True)
        # print("generated_mols",generated_mols[0])
        # Chem.SanitizeMol(generated_mols[0])
        print("final SMILES", generated_mols[0])
        result_SMILES_list.append(generated_mols[0])

        evaluate_SMILES_list(result_SMILES_list)
        print()
