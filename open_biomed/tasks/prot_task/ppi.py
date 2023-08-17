import logging
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
import argparse
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.ppi_dataset import SUPPORTED_PPI_DATASETS
from models.task_model.ppi_model import PPISeqModel, PPIGraphModel
from utils import PPICollator, AverageMeter, EarlyStopping, ToDevice
from utils.metrics import multilabel_f1

def train_ppi(train_loader, train_network, val_loader, val_network, model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    running_loss = AverageMeter()
    stopper = EarlyStopping(mode='higher', patience=args.patience, filename=args.output_path)
    for epoch in range(args.epochs):
        logger.info("Epoch %d" % (epoch))
        logger.info("Training...")
        model.train()
        step = 0
        for prot1, prot2, label in train_loader:
            prot1 = ToDevice(prot1, device)
            prot2 = ToDevice(prot2, device)
            label = label.to(device)
            if train_network is not None:
                pred = model(prot1, prot2, train_network)
            else:
                pred = model(prot1, prot2)
            loss = loss_fn(pred, label)
            loss.backward()
            
            running_loss.update(loss.item(), label.size(0))
            step += 1

            optimizer.step()
            optimizer.zero_grad()
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()

        if val_loader is not None:
            results = eval_ppi(val_loader, val_network, model, args, device)
            logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))
            if stopper.step(results["F1"], model):
                break
    if val_loader is not None:
        model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def eval_ppi(val_loader, val_network, model, args, device):
    model.eval()
    logger.info("Validating...")
    loss_fn = nn.BCEWithLogitsLoss()

    all_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for prot1, prot2, label in val_loader:
            prot1 = ToDevice(prot1, device)
            prot2 = ToDevice(prot2, device)
            label = label.to(device)
            if val_network is not None:
                pred = model(prot1, prot2, val_network)
            else:
                pred = model(prot1, prot2)
            all_loss += loss_fn(pred, label).item()
            pred = torch.sigmoid(pred) > 0.5
            
            all_preds.append(np.array(pred.detach().cpu(), dtype=float))
            all_labels.append(np.array(label.detach().cpu()))

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1, precision, recall = multilabel_f1(all_preds, all_labels)
    results = {
        "F1": f1,
        "Precision": precision,
        "Recall": recall, 
    }
    return results

def main(args, config):
    # configure seed
    random.seed(42)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.graph_ppi = config["model"].startswith("gnn_ppi")
    dataset = SUPPORTED_PPI_DATASETS[args.dataset](args.dataset_path, config["data"]["protein"], directed=False, make_network=args.graph_ppi, split=args.split_strategy)

    if args.mode == "train":
        train_dataset = dataset.index_select(dataset.train_indexes, split="train")
        test_dataset = dataset.index_select(dataset.test_indexes, split="test")
        logger.info("Num train: %d, Num test: %d" % (len(train_dataset), len(test_dataset)))
        collator = PPICollator(config["data"]["protein"], args.graph_ppi)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

        device = torch.device(args.device)
        if not args.graph_ppi:
            model = PPISeqModel(config["network"], train_dataset.num_classes)
            train_network = None
            test_network = None
        else:
            model = PPIGraphModel(config["network"], train_dataset.num_classes)
            train_network = train_dataset.network.to(device)
            test_network = test_dataset.network.to(device)
        if args.init_checkpoint != "None":
            ckpt = torch.load(args.init_checkpoint)
            if args.param_key != "None":
                ckpt = ckpt[args.param_key]
            model.load_state_dict(ckpt)
        model = model.to(device)
        
        model = train_ppi(train_loader, train_network, test_loader, test_network, model, args, device)
        results = eval_ppi(test_loader, test_network, model, args, device)
        logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default="SHS27k")
    parser.add_argument("--dataset_path", type=str, default="../datasets/ppi/SHS27k/")
    parser.add_argument("--split_strategy", type=str, default="random")
    parser.add_argument("--init_checkpoint", type=str, default="None")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/ppi/finetune.pth")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    config = json.load(open(args.config_path, "r"))
    main(args, config)