import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, f1_score

from datasets.ctc_dataset import SUPPORTED_CTC_DATASETS
from models.ctc_model import CTCModel
from utils import EarlyStopping, AverageMeter, seed_all, ToDevice
from utils.ditributed_utils import init_distributed_mode, get_rank, is_main_process, concat_reduce
from utils.schedulars import CosineAnnealingWarmupRestarts

def add_arguments(parser):
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='zheng68k')
    parser.add_argument("--dataset_path", type=str, default='../datasets/ctc/zheng68k/')
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/ctc/finetune.pth")
    
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes') 
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--unassign_threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser

def train_ctc(train_loader, val_loader, model, device, args):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedular = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    stopper = EarlyStopping(mode="higher", patience=args.patience, filename=args.output_path)
    running_loss = AverageMeter(distributed=args.distributed, local_rank=args.local_rank, dest_device=0, world_size=args.world_size)
    running_acc = AverageMeter(distributed=args.distributed, local_rank=args.local_rank, dest_device=0, world_size=args.world_size)
    
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        model.train()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        running_loss.reset()
        running_acc.reset()

        for step, (cell, label) in enumerate(tqdm(train_loader)):
            cell, label = ToDevice(cell, device), label.to(device)
            logits = model(cell)
            loss = loss_fn(logits, label)
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
            pred = nn.Softmax(dim=-1)(logits).argmax(dim=-1)
            correct = torch.eq(pred, label).sum(dim=-1).item()

            running_loss.update(loss.item())
            running_acc.update(correct / args.batch_size)
            if (step + 1) * args.world_size % args.logging_steps == 0:
                if args.distributed:
                    dist.barrier()
                logger.info("Steps=%d Training Loss=%.4lf, Acc=%.4lf" % (
                    step, 
                    running_loss.get_average(),
                    running_acc.get_average()
                ))
                running_loss.reset()
                running_acc.reset()
        if args.distributed:
            dist.barrier()
        schedular.step()
        results = val_ctc(val_loader, model, device, args)
        logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))
        if args.distributed:
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        if stopper.step((results["Accuracy"]), model_without_ddp):
            break
    
    model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def val_ctc(val_loader, model, device, args):
    model.eval()
    all_preds, all_y = [], []
    for cell, label in tqdm(val_loader):
        cell= ToDevice(cell, device)
        logits = model(cell)
        pred = nn.Softmax(dim=-1)(logits).argmax(dim=-1).detach().cpu()
        pred[np.amax(np.array(pred), axis=-1) < args.unassign_threshold] = -1

        all_preds.append(pred)
        all_y.append(label)
    if args.distributed:
        dist.barrier()
        all_preds = concat_reduce(all_preds, len(val_loader.dataset), args.world_size)
        all_y = concat_reduce(all_y, len(val_loader.dataset), args.world_size)
    else:
        all_preds = torch.cat(all_preds, dim=0)
        all_y = torch.cat(all_y, dim=0)
    return {
        "Accuracy": accuracy_score(all_preds, all_y),
        "F1 Score": f1_score(all_preds, all_y),
    }

def main(args, config):
    # prepare dataset
    dataset = SUPPORTED_CTC_DATASETS[args.dataset](args.dataset_path, config["data"], args.seed)

    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    if args.distributed:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset, shuffle=True))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=DistributedSampler(val_dataset, shuffle=False))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # prepare model
    device = torch.device(args.device)
    model = CTCModel(config["network"], dataset.num_classes)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device])

    seed_all(args.seed)

    if args.mode == "train":
        train_ctc(train_loader, val_loader, model, device, args)
        results = val_ctc(val_loader, model, device, args)
        print(results)
    elif args.mode == "test":
        results = val_ctc(val_loader, model, device, args)
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    init_distributed_mode(args)

    # Print INFO on main process and DEBUG and 
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(process)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process() else logging.WARN,
    )

    config = json.load(open(args.config_path, "r"))
    main(args, config)