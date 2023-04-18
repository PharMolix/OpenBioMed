import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import math
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

import torch
import torch.nn.functional as F

from datasets.text2mol_dataset import SUPPORTED_TEXT2MOLGEN_DATASET
from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugGGNNFeaturizer
from models.drug_encoder import MoMu
from models.drug_decoder import MoFlow, construct_mol, check_validity
from utils import AverageMeter, ToDevice

atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]

SUPPORTED_DRUG_ENCODER = {
    "MoMu": MoMu,
}
SUPPORTED_DRUG_DECODER = {
    "MoFlow": MoFlow,
    "MolT5": None
}

def generate_mol(z, decoder, featurizer, device):
    adj, x = decoder.decode(z)
    mol = construct_mol(x.detach().cpu().numpy(), adj.detach().cpu().numpy(), atomic_num_list)
    if featurizer is not None:
        mol = featurizer(mol).to(device)
    atoms = torch.argmax(x, dim=1)
    x = x[atoms != len(atomic_num_list) - 1]
    x = x.softmax(dim=1)
    return mol, x

def optimize_z(z, anchor, text_feat, encoder, decoder, structure_featurizer, args, device):
    optimizer = torch.optim.Adam([z.requires_grad_()], lr=0.01)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, args.optimize_steps, gamma=0.1)

    running_loss_text = AverageMeter()
    running_loss_anchor = AverageMeter()
    for i in range(args.optimize_steps):
        mol, x = generate_mol(z, decoder, structure_featurizer, device)
        if (x.shape[0] < 1) or (x.shape[1] < 5) or (mol.x.shape[0] < 1) or (mol.x.shape[1] < 2):
            logger.warn("x shape: ", x.shape, "mol.x shape: ", mol.x.shape[0], "Too small, exited")
            break
        mol_feat = encoder.encode_structure_with_prob(mol, x, atomic_num_list, device)
        mol_feat = F.normalize(mol_feat, dim=1)
        loss = - mol_feat @ text_feat.t() / 0.1
        running_loss_text.update(loss.detach().cpu().item())
        """
        if anchor is not None:
            loss_anchor = torch.nn.MSELoss(reduction='sum')(z, anchor)
            #loss_anchor = F.sigmoid(- mol_feat @ anchor.t() / 0.1)
            running_loss_anchor.update(loss_anchor.detach().cpu().item())
            loss += args.lambd * loss_anchor
        """
        if torch.isnan(loss):
            logger.warn("loss is nan, exited")
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedular.step()
        
        
        if i % args.logging_steps == 0:
            logger.info("Steps=%d Loss1=%.4lf Loss2=%.4lf" % (i, running_loss_text.get_average(), running_loss_anchor.get_average()))
            running_loss_text.reset()
            running_loss_anchor.reset()
    return z

def stop_gradient(model):
    for key, params in model.named_parameters():
        params.requires_grad = False

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--technique", type=str, default="z_optimize")
    parser.add_argument("--encoder_config_path", type=str, default="")
    parser.add_argument("--init_encoder_checkpoint", type=str, default="None")
    parser.add_argument("--encoder_param_key", type=str, default="None")
    parser.add_argument("--decoder_config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='AttrPrompt')
    parser.add_argument("--dataset_path", type=str, default='../datasets/molgen/attr_prompt')

def add_z_optimize_arguments(parser):
    parser.add_argument("--rounds_per_text", type=int, default=60)
    parser.add_argument("--optimize_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambd", type=float, default=1.0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save_fig", action="store_true")
    parser.add_argument("--save_path", type=str, default="../tmps/molgen/")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    if args.technique == "z_optimize":
        add_z_optimize_arguments(parser)
        args = parser.parse_args()
    encoder_config = json.load(open(args.encoder_config_path, "r"))
    decoder_config = json.load(open(args.decoder_config_path, "r"))

    # load dataset
    dataset = SUPPORTED_TEXT2MOLGEN_DATASET[args.dataset](args.dataset_path, encoder_config["data"]["drug"])
    
    # load featurizer
    feat_config = encoder_config["data"]["drug"]["featurizer"]["structure"]
    structure_featurizer = SUPPORTED_DRUG_FEATURIZER[feat_config["name"]](feat_config)

    # load encoder
    device = torch.device(args.device)
    encoder = SUPPORTED_DRUG_ENCODER[encoder_config["model"]](encoder_config["network"])
    if args.init_encoder_checkpoint != "None":
        ckpt = torch.load(args.init_encoder_checkpoint, map_location="cpu")
        if args.encoder_param_key != "None":
            ckpt = ckpt[args.encoder_param_key]
        encoder.load_state_dict(ckpt)
    
    # load decoder
    decoder = SUPPORTED_DRUG_DECODER[decoder_config["model"]](decoder_config["network"])

    if args.technique == "z_optimize":
        anchor = Chem.MolFromSmiles("COC(=O)C1=C(C)N(C)C(C)=C(C(=O)OC)C1c1ccc(Cl)cc1")
        print("anchor: COC(=O)C1=C(C)N(C)C(C)=C(C(=O)OC)C1c1ccc(Cl)cc1 log_p:", Descriptors.MolLogP(anchor), "TPSA:", Descriptors.TPSA(anchor))
        img = Draw.MolsToGridImage([anchor], legends=['COC(=O)C1=C(C)N(C)C(C)=C(C(=O)OC)C1c1ccc(Cl)cc1'],
                                           molsPerRow=1, subImgSize=(300, 300))
        img.save(os.path.join(args.save_path, 'anchor.png'))

        #anchor = None
        featurizer = DrugGGNNFeaturizer({"max_n_atoms": 38, "atomic_num_list": atomic_num_list})
        x, adj, normalized_adj = featurizer(anchor)
        z_dim = decoder.a_size + decoder.b_size
        mean = torch.zeros(1, z_dim)
        std = torch.ones(1, z_dim) * math.sqrt(math.exp(decoder.ln_var.item())) * args.temperature
        
        encoder.eval()
        decoder.eval()
        stop_gradient(encoder)
        stop_gradient(decoder)
        encoder.to(device)
        decoder.to(device)


        with torch.no_grad():
            x = x.unsqueeze(0).to(device)
            adj = adj.unsqueeze(0).to(device)
            normalized_adj = normalized_adj.unsqueeze(0).to(device)
            z_init, _ = decoder(adj, x, normalized_adj)
            z_init = torch.cat((z_init[0].view(1, -1), z_init[1].view(1, -1)), dim=1)
            mean = z_init.detach().cpu()


        mols = {}
        for i, text in enumerate(dataset.texts):
            text = {k: torch.tensor([v]).to(device) for k, v in text.items()}
            text_feat = F.normalize(encoder.encode_text(text), dim=-1)
            all_adj, all_x = [], []
            valid_ratio, unique_ratio = [], []
            for j in range(args.rounds_per_text):
                z = torch.normal(mean, std).to(device)
                z.requires_grad_(True)
                z = optimize_z(z, anchor, text_feat, encoder, decoder, structure_featurizer, args, device)
                adj, x = decoder.decode(z)
                all_adj.append(adj.detach().cpu().numpy())
                all_x.append(x.detach().cpu().numpy())
            result = check_validity(all_adj, all_x, atomic_num_list)
            valid_ratio.append(result["valid_ratio"])
            unique_ratio.append(result["unique_ratio"])
            mols[i] = result["valid_smiles"]
            if args.save_fig:
                os.makedirs(os.path.join(args.save_path, "text-" + str(i)), exist_ok=True)
                for j, mol in enumerate(result["valid_mols"]):
                    save_path = os.path.join(args.save_path, "text-" + str(i), "mol-" + str(j) + ".png")
                    img = Draw.MolsToGridImage([mol], legends=[result['valid_smiles'][j]],
                                           molsPerRow=1, subImgSize=(300, 300))
                    img.save(save_path)
            if args.evaluate:
                for smi in result["valid_smiles"]:
                    mol = Chem.MolFromSmiles(smi)
                    logger.info("smi: %s, log_p: %.4lf, tPSA: %.4lf" % (smi, Descriptors.MolLogP(mol), Descriptors.TPSA(mol)))

        pickle.dump(mols, open(os.path.join(args.save_path, "smiles.pkl"), "wb"))
        logger.info("Valid ratio %.4lf±%.4lf" % (np.mean(valid_ratio), np.std(valid_ratio)))
        logger.info("Unique ratio %.4lf±%.4lf" % (np.mean(unique_ratio), np.std(unique_ratio)))

    elif args.technique == "adapt":
        pass