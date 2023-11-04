import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from models.MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from models.MoleculeSTM.models import GNN, GNN_graphpred, MLP
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_SMILES_list(args):
    if args.input_SMILES is not None:
        SMILES_list = [args.input_SMILES]
    else:
        SMILES_list = []
        f = open(args.input_SMILES_file, 'r')
        lines = f.readlines()
        for line in lines:
            SMILES = line.strip()
            if len(SMILES) > 0:
                SMILES_list.append(SMILES)
    return SMILES_list


description_dict = {
    101: "This molecule is soluble in water.",
    102: "This molecule is insoluble in water.",
    103: "This molecule is like a drug.",
    104: "This molecule is not like a drug.",
    105: "This molecule has high permeability.",
    106: "This molecule has low permeability.",
    107: "This molecule has more hydrogen bond acceptors.",
    108: "This molecule has more hydrogen bond donors.",
    109: "This molecule has high bioavailability.",
    110: "This molecule has low toxicity.",
    111: "This molecule is metabolically stable.",
    
    201: "This molecule is soluble in water and has more hydrogen bond acceptors.",
    202: "This molecule is insoluble in water and has more hydrogen bond acceptors.",
    203: "This molecule is soluble in water and has more hydrogen bond donors.",
    204: "This molecule is insoluble in water and has more hydrogen bond donors.",
    205: "This molecule is soluble in water and has high permeability.",
    206: "This molecule is soluble in water and has low permeability.",

    301: "This molecule looks like Penicillin.",
    302: "This molecule looks like Aspirin.",
    303: "This molecule looks like Caffeine.",
    304: "This molecule looks like Cholesterol.",
    305: "This molecule looks like Dopamine.",
    306: "This molecule looks like Cysteine.",
    307: "This molecule looks like Glutathione.",
    
    401: "This molecule is tested positive in an assay that are inhibitors and substrates of an enzyme protein. It uses molecular oxygen inserting one oxygen atom into a substrate, and reducing the second into a water molecule.",
    402: "This molecule is tested positive in an assay for Anthrax Lethal, which acts as a protease that cleaves the N-terminal of most dual specificity mitogen-activated protein kinase kinases.",
    403: "This molecule is tested positive in an assay for Activators of ClpP, which cleaves peptides in various proteins in a process that requires ATP hydrolysis and has a limited peptidase activity in the absence of ATP-binding subunits.",
    404: "This molecule is tested positive in an assay for activators involved in the transport of proteins between the endosomes and the trans Golgi network.",
    405: "This molecule is an inhibitor of a protein that prevents the establishment of the cellular antiviral state by inhibiting ubiquitination that triggers antiviral transduction signal and inhibits post-transcriptional processing of cellular pre-mRNA.",
    406: "This molecule is tested positive in the high throughput screening assay to identify inhibitors of the SARS coronavirus 3C-like Protease, which cleaves the C-terminus of replicase polyprotein at 11 sites.",
}


def get_description_list(args):
    if args.input_description is not None:
        description_list = [args.input_description]
    elif args.input_description_id is None:
        raise ValueError
    else:
        print("Use {} descrition.".format(args.input_description_id))
        description_list = [description_dict[args.input_description_id]]
    print("description_list", description_list)
    return description_list


# https://pubchem.ncbi.nlm.nih.gov/compound/5904
# Penicillin_SMILES = "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
Penicillin_SMILES = "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"

# https://pubchem.ncbi.nlm.nih.gov/compound/2244
# Aspirin_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
Aspirin_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# https://pubchem.ncbi.nlm.nih.gov/compound/2519
# Caffeine_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
Caffeine_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

# https://pubchem.ncbi.nlm.nih.gov/compound/5997
# Cholesterol_SMILES = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
Cholesterol_SMILES = "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"

# https://pubchem.ncbi.nlm.nih.gov/compound/681
# Dopamine_SMILES = "C1=CC(=C(C=C1CCN)O)O"
Dopamine_SMILES = "NCCc1ccc(O)c(O)c1"

# https://pubchem.ncbi.nlm.nih.gov/compound/5862
# Cysteine_SMILES = "C(C(C(=O)O)N)S"
Cysteine_SMILES = "NC(CS)C(=O)O"

# https://pubchem.ncbi.nlm.nih.gov/compound/124886
# Glutathione_SMILES = "C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N"
Glutathione_SMILES = "NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O"


def load_molecule_models(args):
    """
    This function returns the two encoders, one for molecule generative model and one for CLIP.
    """
    if args.MoleculeSTM_molecule_type == "SMILES":
        # This is loading from the pretarined_MegaMolBART
        MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.MegaMolBART_generation_model_dir, output_dir=None)
        molecule_model_generation = copy.deepcopy(MegaMolBART_wrapper.model)
        print("Loading from pretrained MegaMolBART ({}).".format(args.MegaMolBART_generation_model_dir))
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

    else:
        # This is loading from the pretarined_MegaMolBART
        MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.MegaMolBART_generation_model_dir, output_dir=None)
        molecule_model_generation = copy.deepcopy(MegaMolBART_wrapper.model)
        print("Loading from pretrained MegaMolBART ({}).".format(args.MegaMolBART_generation_model_dir))
        molecule_dim_generation = 256

        # This is loading GNN from the pretrained_GNN
        molecule_node_model = GNN(num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
        molecule_model_MoleculeSTM = GNN_graphpred(num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling, num_tasks=1, molecule_node_model=molecule_node_model) 
        print("Start from pretrained model (MoleculeSTM) in {}.".format(args.MoleculeSTM_model_dir))
        input_model_path = os.path.join(args.MoleculeSTM_model_dir, "molecule_model.pth")
        state_dict = torch.load(input_model_path, map_location='cpu')
        molecule_model_MoleculeSTM.load_state_dict(state_dict)
        molecule_dim_MoleculeSTM = args.SSL_emb_dim
        
        mol2latent_MoleculeSTM = nn.Linear(300, molecule_dim_MoleculeSTM)
        input_model_path = os.path.join(args.MoleculeSTM_model_dir, "mol2latent_model.pth")
        print("Loading from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        mol2latent_MoleculeSTM.load_state_dict(state_dict)

    return MegaMolBART_wrapper, molecule_model_generation, molecule_dim_generation, \
        molecule_model_MoleculeSTM, mol2latent_MoleculeSTM, molecule_dim_MoleculeSTM


def load_language_molecule_and_edit_models(args):
    pretrained_SciBERT_folder = os.path.join(args.dataset_path, 'pretrained_SciBERT')
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_mode, cache_dir=pretrained_SciBERT_folder)
    text_model = AutoModel.from_pretrained(args.text_mode, cache_dir=pretrained_SciBERT_folder)

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
    MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.MegaMolBART_generation_model_dir, output_dir=None)
    molecule_model = MegaMolBART_wrapper.model
    print("Loading from pretrained MegaMolBART ({}).".format(args.MegaMolBART_generation_model_dir))
    molecule_dim_generation = 256
    if args.MoleculeSTM_molecule_type == "SMILES":  # For MegaMolBART
        molecule_dim_MoleculeSTM = 256
    else:  # For GIN
        molecule_dim_MoleculeSTM = 300

    text2latent = nn.Linear(text_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "text2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text2latent.load_state_dict(state_dict)
    
    mol2latent = nn.Linear(molecule_dim_MoleculeSTM, args.SSL_emb_dim)
    input_model_path = os.path.join(args.MoleculeSTM_model_dir, "mol2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)

    # generation2MoleculeSTM = nn.Linear(molecule_dim_generation, args.SSL_emb_dim)
    generation2MoleculeSTM = MLP(molecule_dim_generation, [args.SSL_emb_dim, args.SSL_emb_dim])
    input_model_path = os.path.join(args.language_edit_model_dir, "generation2foundation_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    generation2MoleculeSTM.load_state_dict(state_dict)

    # MoleculeSTM2generation = nn.Linear(args.SSL_emb_dim, molecule_dim_generation)
    MoleculeSTM2generation = MLP(args.SSL_emb_dim, [molecule_dim_generation, molecule_dim_generation])
    input_model_path = os.path.join(args.language_edit_model_dir, "foundation2generation_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    MoleculeSTM2generation.load_state_dict(state_dict)

    return text_model, text_tokenizer, text_dim, molecule_model, MegaMolBART_wrapper, molecule_dim_generation, text2latent, mol2latent, generation2MoleculeSTM, MoleculeSTM2generation


def clip_loss_for_edit(molecule_repr, text_repr):
    molecule_repr = F.normalize(molecule_repr, dim=-1)
    text_repr = F.normalize(text_repr, dim=-1)

    similarity = -torch.mm(molecule_repr, text_repr.transpose(0, 1))[0]
    return similarity


def get_molecule_similarity(mol_a, mol_b):
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    return sim


def evaluate_SMILES_list(SMILES_list, description):
    print("SMILES_list:", SMILES_list)
    mol_list = []
    for SMILES in SMILES_list:
        mol = Chem.MolFromSmiles(SMILES)
        # Chem.SanitizeMol(mol)
        # print(SMILES, mol)
        if mol is None:
            continue
        mol_list.append(mol)
    print("valid mol list:", len(mol_list))

    if len(mol_list) < 3:
        return [False]

    if "soluble" in description and "insoluble" not in description:
        props = ["MolLogP"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif "insoluble" in description:
        props = ["MolLogP"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule is more like a drug.", "This molecule is like a drug."]:
        props = ["qed"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule is less like a drug.", "This molecule is not like a drug."]:
        props = ["qed"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has higher permeability.", "This molecule has high permeability."]:
        props = ["TPSA"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has lower permeability.", "This molecule has low permeability."]:
        props = ["TPSA"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has higher molecular weight.", "This molecule has high molecular weight."]:
        props = ["MolWt"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has lower molecular weight.", "This molecule has low molecular weight."]:
        props = ["MolWt"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has more hydrogen bond acceptors."]:
        props = ["NumHAcceptors"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has more hydrogen bond donors."]:
        props = ["NumHDonors"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif "penicillin" in description or "Penicillin" in description:
        target_mol = Chem.MolFromSmiles(Penicillin_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between penicillin and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between penicillin and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity:
            answer = [True]
        else:
            answer = [False]

    elif "aspirin" in description or "Aspirin" in description:
        target_mol = Chem.MolFromSmiles(Aspirin_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between aspirin and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between aspirin and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    elif "caffeine" in description or "Caffeine" in description:
        target_mol = Chem.MolFromSmiles(Caffeine_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between caffeine and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between caffeine and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity:
            answer = [True]
        else:
            answer = [False]

    elif "cholesterol" in description or "Cholesterol" in description:
        target_mol = Chem.MolFromSmiles(Cholesterol_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between cholesterol and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between cholesterol and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    elif "dopamine" in description or "Dopamine" in description:
        target_mol = Chem.MolFromSmiles(Dopamine_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between dopamine and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between dopamine and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity:
            answer = [True]
        else:
            answer = [False]

    elif "cysteine" in description or "Cysteine" in description:
        target_mol = Chem.MolFromSmiles(Cysteine_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between cysteine and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between cysteine and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    elif "glutathione" in description or "Glutathione" in description:
        target_mol = Chem.MolFromSmiles(Glutathione_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between glutathione and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between glutathione and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    else:
        print("Not implemented.")
        answer = [False]

    return answer