import numpy as np
import torch


# This is for BERT
def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)


# This is for BERT
def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


# This is for BERT
def prepare_text_tokens(device, description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(tokens_ids).long().to(device)
    masks = torch.Tensor(masks).bool().to(device)
    return tokens_ids, masks

    
def get_molecule_repr_MoleculeSTM(molecule_data, mol2latent=None, molecule_type="SMILES", MegaMolBART_wrapper=None, molecule_model=None):
    if molecule_type == "SMILES":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding(molecule_data)  # [pad, B, d], [pad, B]
        molecule_repr = embedding[0, :, :]  # [B, d]
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    
    if mol2latent is not None:
        molecule_repr = mol2latent(molecule_repr)
    return molecule_repr


def freeze_network(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def get_num_task_and_type(dataset):
    if dataset in ["esol", "freesolv", "lipophilicity"]:
        return 1, "regression"
    elif dataset in ["hiv", "bace", "bbbp"]:
        return 1, "classification"
    elif dataset == "tox21":
        return 12, "classification"
    elif dataset == "pcba":
        return 92, "classification"
    elif dataset == "muv":
        return 17, "classification"
    elif dataset == "toxcast":
        return 617, "classification"
    elif dataset == "sider":
        return 27, "classification"
    elif dataset == "clintox":
        return 2, "classification"
    raise ValueError("Invalid dataset name.")
