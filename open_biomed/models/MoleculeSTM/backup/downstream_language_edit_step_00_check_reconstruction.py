import argparse
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

import torch
from torch.utils.data import DataLoader as torch_DataLoader

from MoleculeSTM.utils import freeze_network
from MoleculeSTM.datasets import ZINC15_Datasets_Only_SMILES, PubChem_Datasets_Only_SMILES
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART

props = [
    "qed", "MolWt", "MolLogP", "TPSA",
    "HeavyAtomCount", "NumAromaticRings", "NumHAcceptors", "NumHDonors",  "NumRotatableBonds"
]
props = [
    "MolWt", "MolLogP"
]
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--dataspace_path", type=str, default="../../Datasets")
    parser.add_argument("--dataset", type=str, default="ZINC15")
    parser.add_argument("--molecule_type", type=str, default="MegaMolBART", choices=["MegaMolBART", "Graph"])

    ########## for MoleculeSTM ##########
    parser.add_argument("--CLIP_input_model_dir", type=str, default="../../pretrained_model")
    parser.add_argument("--SSL_emb_dim", type=int, default=256)

    ########## for generation ##########
    parser.add_argument("--generation_model_dir", type=str, default="../../Datasets/pretrained_MegaMolBART/checkpoints")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    print(args)

    # This is loading from the pretarined_MegaMolBART
    MegaMolBART_wrapper = MegaMolBART(input_dir=args.generation_model_dir, output_dir=None)
    molecule_model_generation = MegaMolBART_wrapper.model
    print("Loading from pretrained MegaMolBART ({}).".format(args.generation_model_dir))
    molecule_dim_generation = 256

    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    molecule_model_generation = molecule_model_generation.to(device)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    freeze_network(molecule_model_generation)
    molecule_model_generation.eval()

    if args.molecule_type == "MegaMolBART":
        if args.dataset == "ZINC15":
            dataset_root = os.path.join(args.dataspace_path, "ZINC15_data")
            dataset = ZINC15_Datasets_Only_SMILES(dataset_root)
        elif "PubChem" in args.dataset:
            dataset_root = os.path.join(args.dataspace_path, "PubChem_data")
            dataset = PubChem_Datasets_Only_SMILES(dataset_root)
        else:
            raise Exception
        dataloader_class = torch_DataLoader
    else:
        raise Exception
    
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for batch_idx, batch in enumerate(dataloader):
        SMILES_list = batch
        print("SMILES_list", SMILES_list)
        
        for original_SMILES in SMILES_list:
            mol = Chem.MolFromSmiles(original_SMILES)
            for name, func in prop_pred:
                value = func(mol)
                print("{}: {}".format(name, value))
            canon_original_SMILES = Chem.MolToSmiles(mol)

            latent_code_init, pad_mask_init = MegaMolBART_wrapper.smileslist2embedding_model_given(molecule_model_generation, [original_SMILES])  # [pad, B, d], [pad, B]
            print("latent_code:\t", latent_code_init[0, :, :5])

            latent_code_init, pad_mask_init = MegaMolBART_wrapper.smileslist2embedding_model_given(molecule_model_generation, [canon_original_SMILES])  # [pad, B, d], [pad, B]
            print("latent_code:\t", latent_code_init[0, :, :5])

            generated_SMILES = MegaMolBART_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)
            print("original SMILES:          \t", original_SMILES)
            print("original SMILES (canon):  \t", canon_original_SMILES)
            print("reconstructured SMILES:   \t", generated_SMILES[0])
            print()

        if batch_idx >= 9:
            break
