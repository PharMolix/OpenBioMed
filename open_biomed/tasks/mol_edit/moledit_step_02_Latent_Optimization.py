import argparse
import math
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import re
import copy
from utils.molstm_utils import prepare_text_tokens, get_SMILES_list, get_description_list, load_language_molecule_and_edit_models, clip_loss_for_edit, evaluate_SMILES_list
from models.multimodal.moleculestm import MLP
from transformers import BertTokenizer
from models.task_model.moledit_model import MoleditModel
from models.multimodal.mega_molbart.mega_mol_bart import MegaMolBART

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def check_edit(SMILES, text):
    text_list = text_tokenizer(text, truncation=True, padding=True, return_tensors='pt').to(device)
    del text_list["token_type_ids"]
    text_output = text_model(text_list)
    text_repr = text_output


    first_and_second_SMILES_list = []

    latent_code_init, pad_mask_init = MegaMolBART_wrapper.smileslist2embedding([SMILES])  # [pad, B, d], [pad, B]
    first_and_second_SMILES_list.append(SMILES)

    regenerated_mols = MegaMolBART_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)
    first_and_second_SMILES_list.append(regenerated_mols[0])

    l2_lambda_list = [
        1e1, 1e0, 1e-1, 1e-2, 1e-3
    ]
    result_SMILES_list_one_pair, result_eval_list_one_pair = [], []
    
    if args.use_noise_for_init:
        print("Use random noise for init")
        random_noise = torch.randn(latent_code_init.size()).to(device)
    
    for l2_lambda in l2_lambda_list:
        print("l2 lambda: {}".format(l2_lambda))
        current_SMILES_list = [first_and_second_SMILES_list[0]] + [first_and_second_SMILES_list[1]]
        if args.use_noise_for_init:
            print("Use random noise for init")
            latent = latent_code_init.detach().clone() + random_noise
        else:
            print("No random noise for init")
            latent = latent_code_init.detach().clone()
        pad_mask = pad_mask_init.detach().clone()
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

            molecule_repr_generation = mean_pooling(latent, pad_mask) # [B, d]
            if args.normalize:
                molecule_repr_generation = F.normalize(molecule_repr_generation, dim=-1)
            molecule_repr_MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)

            clip_loss_ = clip_loss_for_edit(molecule_repr_MoleculeSTM, text_repr)
            l2_loss_ =  l2_lambda * ((latent_code_init - latent) ** 2).mean()

            loss = clip_loss_ + l2_loss_

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print("clip loss: {:.5f}\tL2 loss: {:.5f}".format(clip_loss_.item(), l2_loss_.item()))

        generated_mols = MegaMolBART_wrapper.inverse_transform([latent], pad_mask.bool().cuda(), k=1, sanitize=True)
        current_SMILES_list.append(generated_mols[0])
        result_SMILES_list_one_pair.append([text] + current_SMILES_list + ['{}'.format(l2_lambda)])

        current_result_list = evaluate_SMILES_list(current_SMILES_list, text)
        result_eval_list_one_pair.append(current_result_list)
        print()
    
    result_eval_list_one_pair = np.array(result_eval_list_one_pair)
    result_eval_list_one_pair = np.any(result_eval_list_one_pair, axis=0, keepdims=True)
    print("result_eval_list_one_pair\n", result_eval_list_one_pair)
    return result_SMILES_list_one_pair, result_eval_list_one_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)

    ########## for editing ##########
    parser.add_argument("--input_description", type=str, default=None)
    parser.add_argument("--input_description_id", type=int, default=101)
    parser.add_argument("--input_SMILES", type=str, default=None)
    parser.add_argument("--input_SMILES_file", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--use_noise_for_init", dest="use_noise_for_init", action="store_true")
    parser.add_argument("--no_noise_for_init", dest="use_noise_for_init", action="store_false")
    parser.set_defaults(use_noise_for_init=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--config_path", type=str, default=None)
    ########## for MoleculeSTM ##########
    parser.add_argument("--MoleculeSTM_molecule_type", type=str, default=None, choices=["SMILES", "Graph"])

    ########## for MegaMolBART ##########
    parser.add_argument("--MegaMolBART_generation_model_dir", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--text_mode", type=str, default=None)

    ########## for MoleculeSTM and generation projection ##########
    parser.add_argument("--language_edit_model_dir", type=str, default=None)   
    ########## for editing ##########
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--MASTER_PORT", type=str, default='6000')
    args = parser.parse_args()

    print(args)

    config = json.load(open(args.config_path)) 
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    device = torch.device(args.device) \
        if torch.cuda.is_available() else torch.device("cpu")
    

    # load model
    text_model = MoleditModel(config["network"])
    text_tokenizer = BertTokenizer.from_pretrained(args.text_mode, model_max_length=512, cache_dir=args.text_mode)
    if config["model"]== "molstm-MegaMolBART":
        MegaMolBART_wrapper = text_model.model.MegaMolBART_wrapper
        molecule_model = MegaMolBART_wrapper.model
    else: 
        MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.MegaMolBART_generation_model_dir, output_dir=None)
        molecule_model = MegaMolBART_wrapper.model
   
    torch.cuda.set_device(int(re.search(r'\d+', args.device).group()))

    generation2MoleculeSTM = MLP(256, [args.SSL_emb_dim, args.SSL_emb_dim])
    input_model_path = os.path.join(args.language_edit_model_dir, "generation2MoleculeSTM_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    generation2MoleculeSTM.load_state_dict(state_dict)


    MoleculeSTM2generation = MLP(args.SSL_emb_dim, [256, 256])
    input_model_path = os.path.join(args.language_edit_model_dir, "MoleculeSTM2generation_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    MoleculeSTM2generation.load_state_dict(state_dict)

    text_model = text_model.to(device)
    molecule_model = molecule_model.to(device)
    generation2MoleculeSTM.to(device)
    MoleculeSTM2generation.to(device)
    text_model.eval()
    molecule_model.eval()
    generation2MoleculeSTM.eval()
    MoleculeSTM2generation.eval()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device) \
        if torch.cuda.is_available() else torch.device("cpu")
    
    print("\n\n\nstart editing\n\n\n")

    source_SMILES_list = get_SMILES_list(args)[0:1]
    description_list = get_description_list(args)



    for description in description_list:
        print("===== for description {} =====".format(description))
        result_SMILES_list, result_acc_list = [], []
        for SMILES in source_SMILES_list:
            print("===== for SMILES {} =====".format(SMILES))
            result_SMILES_list_, result_acc_list_ = check_edit(SMILES, description)
            result_SMILES_list.extend(result_SMILES_list_)
            result_acc_list.append(result_acc_list_)
            print("\n\n\n")
        
        result_acc_list = np.concatenate(result_acc_list, axis=0)
        result_acc_list = np.sum(result_acc_list, axis=0)
        result_acc_list = 100. * result_acc_list / len(source_SMILES_list)
        result_acc_row = '\t'.join(['{}'.format(x) for x in result_acc_list])
        print("===== Accuracy =====\t{}".format(result_acc_row))

        if args.output_model_dir is not None:
            saver_file = os.path.join(args.output_model_dir, "edited_SMILES.tsv")
            with open(saver_file, 'a') as f:
                for row in result_SMILES_list:
                    row = "\t".join(row)
                    print(row, file=f)

            saver_file = os.path.join(args.output_model_dir, "accuracy")
            np.savez(saver_file, result_acc_list)
