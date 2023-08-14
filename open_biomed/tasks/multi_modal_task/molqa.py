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
from transformers import BertTokenizerFast

from datasets.molqa_dataset import SUPPORTED_MOLQA_DATASET
from models.task_model.molqa_model import SUPPORTED_MOLQA_MODELS

from utils import AverageMeter, ToDevice, MolQACollator

def normalize_text(s, rm_punc=True):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if rm_punc:
        s = remove_punc(lower(s))
    else:
        s = lower(s)
    return white_space_fix(remove_articles(s))

def train_molqa(train_loader, test_loader, test_dataset, model, args, device):
    optimizer_grouped_parameters = [p for n, p in list(model.named_parameters()) if not "mol_encoder" in n and not "mol_proj" in n]
    optimizer = torch.optim.Adam([p for p in model.parameters()], lr=args.lr, weight_decay=args.weight_decay)
    #optimizer1 = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    #optimizer2 = torch.optim.Adam([p for p in model.mol_encoder.parameters()] + [p for p in model.mol_proj.parameters()], lr=args.lr*10, weight_decay=args.weight_decay)

    running_loss = AverageMeter()
    step = 0
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        model.train()

        for mol, question, answer in train_loader:
            mol = ToDevice(mol, device)
            question = ToDevice(question, device)
            answer = ToDevice(answer, device)
            loss = model(mol, question, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()
        if (epoch + 1) % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
            print(test_molqa(test_loader, model, args, device))
    return model

def test_molqa(test_loader, model, args, device):
    model.eval()

    exact, f1 = [], []
    logger.info("Testing...")
    with torch.no_grad():
        for i, (mol, question, answer) in enumerate(tqdm(test_loader)):
            mol = ToDevice(mol, device)
            question = ToDevice(question, device)
            output = model.generate(mol, question, num_beams=5, max_length=512)
            if i <= 3:
                logger.info("Outputs: %s" % output)
                logger.info("Ground truth: %s" % answer)
                logger.info("------------------------------------------------------")
            for j in range(len(output)):
                y_true = normalize_text(answer[j], rm_punc=True)
                y_pred = normalize_text(output[j], rm_punc=True)
                exact.append(int(y_true == y_pred))

                y_true = y_true.split()
                y_pred = y_pred.split()
                common_tokens = set(y_true) & set(y_pred)
                if len(common_tokens) == 0:
                    f1.append(0)
                else:
                    precision = len(common_tokens) / len(y_pred)
                    recall = len(common_tokens) / len(y_true)
                    f1.append(2 * precision * recall / (precision + recall))

    return {
        "exact": np.mean(exact),
        "f1": np.mean(f1),
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
    train_dataset = SUPPORTED_MOLQA_DATASET[args.dataset](args.dataset_path, config["data"], split="train")
    test_dataset = SUPPORTED_MOLQA_DATASET[args.dataset](args.dataset_path, config["data"], split="test")
    train_collator = MolQACollator(config["data"], collate_outputs=True)
    test_collator = MolQACollator(config["data"], collate_outputs=False)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=test_collator, num_workers=args.num_workers)

    # load model
    model = SUPPORTED_MOLQA_MODELS[config["network"]["type"]](config["network"])
    if args.init_checkpoint != "None":
        state_dict = torch.load(args.init_checkpoint)
        if args.param_key != "None":
            state_dict = state_dict[args.param_key]
        model.load_state_dict(state_dict)
    model = model.to(device)

    if args.mode == "train":
        train_molqa(train_dataloader, test_dataloader, test_dataset, model, args, device)
    elif args.mode == "test":
        if os.path.exists(args.output_path):
            state_dict = torch.load(args.output_path, map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
        results = test_molqa(test_dataloader, model, args, device)
        print(results)
    elif args.mode == "traintest":
        train_molqa(train_dataloader, test_dataloader, test_dataset, model, args, device)
        results = test_molqa(test_dataloader, model, args, device)
        print(results)
