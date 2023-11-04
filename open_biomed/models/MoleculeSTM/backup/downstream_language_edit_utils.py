import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def load_molecule_models(args):
    """
    This function returns the two encoders, one for molecule generative model and one for CLIP.
    TODO: now we adopt MegaMolBART for both. Will make this more flexible in the future.
    """
    # This is loading from the pretarined_MegaMolBART
    MegaMolBART_wrapper = MegaMolBART(input_dir=args.generation_model_dir, output_dir=None)
    molecule_model_generation = copy.deepcopy(MegaMolBART_wrapper.model)
    print("Loading from pretrained MegaMolBART ({}).".format(args.generation_model_dir))
    molecule_dim_generation = 256
    
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "molecule_model.pth")
    molecule_model_MoleculeSTM = MegaMolBART_wrapper.model
    state_dict = torch.load(input_model_path, map_location='cpu')
    print("Loading from {}...".format(input_model_path))
    molecule_model_MoleculeSTM.load_state_dict(state_dict)
    molecule_dim_MoleculeSTM = args.SSL_emb_dim
    
    mol2latent_MoleculeSTM = nn.Linear(256, molecule_dim_MoleculeSTM)
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "mol2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent_MoleculeSTM.load_state_dict(state_dict)
    return MegaMolBART_wrapper, molecule_model_generation, molecule_dim_generation, \
        molecule_model_MoleculeSTM, mol2latent_MoleculeSTM, molecule_dim_MoleculeSTM


def load_language_molecule_and_edit_models(args):
    pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'pretrained_SciBERT')
    # text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
    # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
    # text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
    text_tokenizer = AutoTokenizer.from_pretrained('/mnt/cyz_dair/projects/MoleculeSTM-main/MoleculeSTM-main/data/pretrained_SciBERT', cache_dir=pretrained_SciBERT_folder)
    text_model = AutoModel.from_pretrained('/mnt/cyz_dair/projects/MoleculeSTM-main/MoleculeSTM-main/data/pretrained_SciBERT', cache_dir=pretrained_SciBERT_folder)

    text_dim = 768

    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "text_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text_model.load_state_dict(state_dict)

    """
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "molecule_model.pth")
    print("Loading from {}...".format(input_model_path))
    MegaMolBART_wrapper = MegaMolBART(input_dir=None, output_dir=None)
    molecule_model = MegaMolBART_wrapper.model
    state_dict = torch.load(input_model_path, map_location='cpu')
    molecule_model.load_state_dict(state_dict)
    """
    # This is loading from the pretarined_MegaMolBART
    MegaMolBART_wrapper = MegaMolBART(input_dir=args.generation_model_dir, output_dir=None)
    molecule_model = MegaMolBART_wrapper.model
    print("Loading from pretrained MegaMolBART ({}).".format(args.generation_model_dir))
    molecule_dim_generation = 256
    molecule_dim_MoleculeSTM = 256

    text2latent = nn.Linear(text_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "text2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text2latent.load_state_dict(state_dict)
    
    mol2latent = nn.Linear(molecule_dim_generation, args.SSL_emb_dim)
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "mol2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)

    generation2MoleculeSTM = nn.Linear(molecule_dim_generation, molecule_dim_MoleculeSTM)
    input_model_path = os.path.join(args.language_edit_model_dir, "generation2MoleculeSTM_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    generation2MoleculeSTM.load_state_dict(state_dict)

    MoleculeSTM2generation = nn.Linear(molecule_dim_MoleculeSTM, molecule_dim_generation)
    input_model_path = os.path.join(args.language_edit_model_dir, "MoleculeSTM2generation_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    MoleculeSTM2generation.load_state_dict(state_dict)

    return text_model, text_tokenizer, text_dim, molecule_model, MegaMolBART_wrapper, molecule_dim_generation, text2latent, mol2latent, generation2MoleculeSTM, MoleculeSTM2generation


def clip_loss_for_edit(molecule_repr, mol2latent, text_repr):
    # molecule_repr = F.normalize(molecule_repr, dim=-1)
    # molecule_repr = mol2latent(molecule_repr)
    molecule_repr = F.normalize(molecule_repr, dim=-1)
    
    text_repr = F.normalize(text_repr, dim=-1)

    similarity = -torch.mm(molecule_repr, text_repr.transpose(0, 1))[0]
    return similarity


def evaluate_SMILES_list(SMILES_list):
    print("SMILES_list:")
    print(SMILES_list)
    mol_list = []
    for SMILES in SMILES_list:
        mol = Chem.MolFromSmiles(SMILES)
        # Chem.SanitizeMol(mol)
        # print(SMILES, mol)
        if mol is None:
            continue
        mol_list.append(mol)
    print("mol_list", len(mol_list))

    print()    
    props = ["MolWt", "MolLogP", "TPSA", "qed"]
    props = ["MolLogP"]
    prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
    for name, func in prop_pred:
        print("evaluating with {}".format(name))
        for SMILES, mol in zip(SMILES_list, mol_list):
            value = func(mol)
            print("====={} & {:.5f}".format(SMILES, value))
        print()

    return