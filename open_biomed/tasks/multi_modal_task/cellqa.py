import logging
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from open_biomed.datasets.cellqa_dataset import SUPPORTED_CELLQA_DATASET
from open_biomed.models.task_model.cellqa_model import SUPPORTED_CELLQA_MODELS

from utils import AverageMeter, ToDevice, CellQACollator

def train_cellqa(train_loader, val_loader, test_loader, model, args, device):
    optimizer = torch.optim.Adam([p for p in model.parameters()], lr=args.lr, weight_decay=args.weight_decay)

    running_loss = AverageMeter()
    step = 0
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        model.train()

        for cell, question, answer in train_loader:
            cell = ToDevice(cell, device)
            question = ToDevice(question, device)
            answer = ToDevice(answer, device)
            loss = model(cell, question, answer)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()
        if (epoch + 1) % args.eval_epochs == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
            print(test_cellqa(test_loader, model, args, device))
    return model

def test_cellqa(test_loader, model, args, device):
    model.eval()

    logger.info("Testing...")
    output_tokens, gt_tokens = [], []
    meteor_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    with torch.no_grad():
        for i, (cell, question, answer) in enumerate(tqdm(test_loader)):
            cell = ToDevice(cell, device)
            question = ToDevice(question, device)
            output = model.generate(cell, question, num_beams=5, max_length=512)
            output = model.decoder_tokenizer.batch_decode(output, skip_special_tokens=True)
            if i <= 3:
                logger.info("Outputs: %s" % output)
                logger.info("Ground truth: %s" % answer)
                logger.info("------------------------------------------------------")
            for j in range(len(output)):
                output_tokens.append(output[j].split(" "))
                gt_tokens.append([answer[j].split(" ")])
                meteor_scores.append(meteor_score(gt_tokens[-1], output_tokens[-1]))
                rouge_scores.append(scorer.score(output[j], answer[j]))

    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
    }

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='chembl-qa')
    parser.add_argument("--dataset_path", type=str, default='./datasets/molqa/ChEMBL')
    parser.add_argument("--init_checkpoint", type=str, default='None')
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--output_path", type=str, default='./ckpts/finetune_ckpts/molqa/molt5')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=300)

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
    train_dataset = SUPPORTED_CELLQA_DATASET[args.dataset](args.dataset_path, config["data"], split="train")
    val_dataset = SUPPORTED_CELLQA_DATASET[args.dataset](args.dataset_path, config["data"], split="valid")
    test_dataset = SUPPORTED_CELLQA_DATASET[args.dataset](args.dataset_path, config["data"], split="test")
    train_collator = CellQACollator(config["data"], collate_outputs=True)
    test_collator = CellQACollator(config["data"], collate_outputs=False)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_collator, num_workers=args.num_workers)
    val_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=train_collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=test_collator, num_workers=args.num_workers)

    # load model
    model = SUPPORTED_CELLQA_MODELS[config["network"]["type"]](config["network"])
    if args.init_checkpoint != "None":
        state_dict = torch.load(args.init_checkpoint)
        if args.param_key != "None":
            state_dict = state_dict[args.param_key]
        model.load_state_dict(state_dict)
    model = model.to(device)

    if args.mode == "train":
        train_cellqa(train_dataloader, val_dataloader, test_dataloader, model, args, device)
    elif args.mode == "test":
        if os.path.exists(args.output_path):
            state_dict = torch.load(args.output_path, map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
        results = test_cellqa(test_dataloader, model, args, device)
        print(results)
    elif args.mode == "traintest":
        train_cellqa(train_dataloader, val_dataloader, test_dataloader, model, args, device)
        results = test_cellqa(test_dataloader, model, args, device)
        print(results)
