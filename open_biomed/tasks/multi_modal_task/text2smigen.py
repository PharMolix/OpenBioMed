import logging
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
RDLogger.DisableLog('rdApp.*')
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev

from datasets.molcap_dataset import SUPPORTED_MOLCAP_DATASET
from models.drug_encoder import Text2MolMLP
from models.text2smi_model import Text2SMILESModel

from utils import AverageMeter, ToDevice, DrugCollator

def train_text2smi(train_loader, val_loader, test_loader, test_dataset, model, args, device):
    requires_grad = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            requires_grad.append(k)
    logger.info("parameters requires grad: %s" % (" ".join(requires_grad)))

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    running_loss = AverageMeter()
    step = 0
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()

        for mol in train_loader:
            mol = ToDevice(mol, device)
            loss = model(mol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()
        val_text2smi(val_loader, model, device)
        if (epoch + 1) % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
            print(test_text2smi(test_dataset, test_loader, model, args, device))
    return model

def val_text2smi(val_loader, model, device):
    model.eval()
    val_loss = 0

    logger.info("Validating...")
    for mol in val_loader:
        mol = ToDevice(mol, device)
        loss = model(mol)
        val_loss += loss.detach().cpu().item()
    logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def test_text2smi(test_dataset, test_loader, model, args, device):
    model.eval()
    outputs = []
    gts = test_dataset.smiles

    logger.info("Testing...")
    for i, mol in enumerate(tqdm(test_loader)):
        mol = ToDevice(mol, device)
        output = model.decode(mol, num_beams=5, max_length=512)
        outputs += output
        if i <= 3:
            for j in range(5):
                logger.info("Generated:    %s" % outputs[-j])
                logger.info("Ground truth: %s" % gts[len(outputs) - j])
                logger.info("------------------------------------------------------")

    N = len(outputs)
    output_tokens = []
    gt_tokens = []
    levs = []
    maccs_sim, rdk_sim, morgan_sim = [], [], []
    n_bad_mols = 0
    n_exact = 0
    with open(args.smi_save_path, "w") as f:
        f.write("text\tground truth\toutput\n")
        for i in range(N):
            output_tokens.append([c for c in outputs[i]])
            gt_tokens.append([[c for c in gts[i]]])
            try:
                mol_output = Chem.MolFromSmiles(outputs[i])
                mol_gt = Chem.MolFromSmiles(gts[i])
                if Chem.MolToInchi(mol_output) == Chem.MolToInchi(mol_gt):
                    n_exact += 1
                maccs_sim.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol_output), MACCSkeys.GenMACCSKeys(mol_gt), metric=DataStructs.TanimotoSimilarity))
                rdk_sim.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_output), Chem.RDKFingerprint(mol_gt), metric=DataStructs.TanimotoSimilarity))
                morgan_sim.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_output, 2), AllChem.GetMorganFingerprint(mol_gt, 2)))
            except:
                n_bad_mols += 1
            levs.append(lev(outputs[i], gts[i]))
            f.write("%s\t%s\t%s\n" % (test_dataset.texts[i], gts[i], outputs[i]))

    bleu = corpus_bleu(gt_tokens, output_tokens)
    return {
        "BLEU": bleu,
        "Levenshtein": np.mean(levs),
        "Valid": 1 - n_bad_mols * 1.0 / N,
        "Exact": n_exact * 1.0 / N,
        "MACCS FTS": np.mean(maccs_sim),
        "RDKit FTS": np.mean(rdk_sim),
        "Morgan FTS": np.mean(morgan_sim),
    }

def test_text2mol(args):
    text2mol = Text2MolMLP(
        ninp=768, 
        nhid=600, 
        nout=300, 
        model_name_or_path=args.text2mol_bert_path, 
        cid2smiles_path=None,
        cid2vec_path=None,
        mol2vec_output_path=os.path.join(args.text2mol_data_path, "tmp.csv")
    )
    text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path))
    device = torch.device(args.device)
    text2mol.to(device)

    logger.info("Calculating Text2Mol Metric...")
    text2mol_scores = []
    bad_smiles = 0
    with open(args.smi_save_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            line = line.rstrip("\n").split("\t")
            try:
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(line[2]))
                if smi in text2mol.smiles2vec:
                    text2mol_scores.append(text2mol(smi, line[0], device).detach().cpu().item())
            except:
                bad_smiles += 1
    logger.info("Bad SMILES: %d" % (bad_smiles))
    return np.mean(text2mol_scores)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='chebi-20')
    parser.add_argument("--dataset_path", type=str, default='../datasets/molcap/chebi-20')
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/text2smi/")
    parser.add_argument("--smi_save_path", type=str, default="../assets/outputs.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=300)
    parser.add_argument("--text2mol_bert_path", type=str, default="")
    parser.add_argument("--text2mol_data_path", type=str, default="")
    parser.add_argument("--text2mol_ckpt_path", type=str, default="")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    device = torch.device(args.device)

    if args.mode == "test_text2mol":
        print("Text2Mol:", test_text2mol(args))
        exit(0)

    # load dataset
    train_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="train")
    val_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="validation")
    test_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="test")
    collator = DrugCollator(config["data"]["drug"])
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    # load model
    model = Text2SMILESModel(config["network"])
    model = model.to(device)

    if args.mode == "train":
        train_text2smi(train_dataloader, val_dataloader, test_dataloader, test_dataset, model, args, device)
    elif args.mode == "test":
        if os.path.exists(args.output_path):
            state_dict = torch.load(args.output_path, map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
        results = test_text2smi(test_dataset, test_dataloader, model, args, device)
        print(results)
    elif args.mode == "traintest":
        train_text2smi(train_dataloader, val_dataloader, test_dataloader, test_dataset, model, args, device)
        results = test_text2smi(test_dataset, test_dataloader, model, args, device)
        print(results)