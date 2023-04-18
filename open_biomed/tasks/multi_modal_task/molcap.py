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
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from datasets.molcap_dataset import SUPPORTED_MOLCAP_DATASET
from models.drug_encoder import Text2MolMLP
from models.molcap_model import MolCapModel, GraphEnhancedMolCapModel

from utils import AverageMeter, ToDevice, DrugCollator

def train_molcap(train_loader, val_loader, test_loader, test_dataset, model, args, device):
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
        val_molcap(val_loader, model, device)
        if (epoch + 1) % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
            print(test_molcap(test_dataset, test_loader, model, args, device))
    return model

def val_molcap(val_loader, model, device):
    model.eval()
    val_loss = 0

    logger.info("Validating...")
    with torch.no_grad():
        for mol in val_loader:
            mol = ToDevice(mol, device)
            loss = model(mol)
            val_loss += loss.detach().cpu().item()
    logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def test_molcap(test_dataset, test_loader, model, args, device):
    model.eval()
    outputs = []
    gts = test_dataset.texts

    logger.info("Testing...")
    with torch.no_grad():
        for i, mol in enumerate(tqdm(test_loader)):
            mol = ToDevice(mol, device)
            output = model.decode(mol, num_beams=5, max_length=512)
            outputs += output
            if i <= 3:
                for j in range(5):
                    logger.info("Generated: %s" % outputs[-j])
                    logger.info("Ground truth: %s" % gts[len(outputs) - j])
                    logger.info("------------------------------------------------------")

    tokenizer = BertTokenizerFast.from_pretrained(args.text2mol_bert_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    text2mol_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    text2mol = Text2MolMLP(
        ninp=768, 
        nhid=600, 
        nout=300, 
        model_name_or_path=args.text2mol_bert_path, 
        cid2smiles_path=os.path.join(args.text2mol_data_path, "cid_to_smiles.pkl"),
        cid2vec_path=os.path.join(args.text2mol_data_path, "test.txt")
    )
    text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path))
    device = torch.device(args.device)
    text2mol.to(device)
    with open(args.caption_save_path, "w") as f:
        f.write("SMILES\tground truth\toutput\n")
        for i in range(len(outputs)):
            output_tokens.append(tokenizer.tokenize(outputs[i], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(gts[i], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
            rouge_scores.append(scorer.score(outputs[i], gts[i]))
            text2mol_scores.append(text2mol(test_dataset.smiles[i], outputs[i], device).detach().cpu().item())
            f.write(test_dataset.smiles[i] + '\t' + gts[i] + '\t' + outputs[i] + '\n')
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
        "Text2Mol": np.mean(text2mol_scores)
    }

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='chebi-20')
    parser.add_argument("--dataset_path", type=str, default='../datasets/molcap/chebi-20')
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/caption.pth")
    parser.add_argument("--caption_save_path", type=str, default="../assets/outputs.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=300)
    parser.add_argument("--text2mol_bert_path", type=str, default="../ckpts/text_ckpts/scibert_scivocab_uncased/")
    parser.add_argument("--text2mol_data_path", type=str, default="../assets/molcap/text2mol_data/")
    parser.add_argument("--text2mol_ckpt_path", type=str, default="../ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt")

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

    # load dataset
    train_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="train")
    val_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="validation")
    test_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"]["drug"], split="test")
    collator = DrugCollator(config["data"]["drug"])
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    # load model
    if config["data"]["drug"]["featurizer"]["structure"]["name"] == "MultiScale":
        model = GraphEnhancedMolCapModel(config["network"])
    else:
        model = MolCapModel(config["network"])
    model = model.to(device)

    if args.mode == "train":
        train_molcap(train_dataloader, val_dataloader, test_dataloader, test_dataset, model, args, device)
    elif args.mode == "test":
        if os.path.exists(args.output_path):
            state_dict = torch.load(args.output_path, map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
        results = test_molcap(test_dataset, test_dataloader, model, args, device)
        print(results)
    elif args.mode == "traintest":
        train_molcap(train_dataloader, val_dataloader, test_dataloader, test_dataset, model, args, device)
        results = test_molcap(test_dataset, test_dataloader, model, args, device)
        print(results)
