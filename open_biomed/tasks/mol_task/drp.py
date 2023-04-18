import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import copy
import math
import numpy as np
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from datasets.drp_dataset import SUPPORTED_DRP_DATASET, TCGA
from models.drp_model import TGDRP
from utils import EarlyStopping, AverageMeter, seed_all, roc_auc, metrics_average
from utils.collators import DRPCollator

SUPPORTED_DRP_MODEL = {
    "TGDRP": TGDRP,
}

def train_drp(train_loader, val_loader, model, args):
    if args.task == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
        metric = "roc_auc"
        mode = "higher"
    elif args.task == "regression":
        loss_fn = nn.MSELoss()
        metric = "rmse"
        mode = "lower"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(mode=mode, patience=args.patience)

    running_loss = AverageMeter()
    for i in range(args.epochs):
        logger.info("========Epoch %d========" % (i + 1))
        logger.info("Training...")
        model.train()
        step_loss = 0
        running_loss.reset()
        t = tqdm(train_loader, desc="Loss=%.4lf" % (step_loss))
        for drug, cell, label in t:
            if isinstance(cell, list):
                drug, cell, label = drug.to(args.device), [feat.to(args.device) for feat in cell], label.to(args.device)
            else:
                drug, cell, label = drug.to(args.device), cell.to(args.device), label.to(args.device)
            pred = model(drug, cell)
            loss = loss_fn(pred, label.view(-1, 1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            running_loss.update(step_loss)
            t.set_description("Loss=%.4lf" % (step_loss))
        logger.info("Average training loss %.4lf" % (running_loss.get_average()))

        val_metrics = val_drp(val_loader, model, args)
        if stopper.step(val_metrics[metric], model):
            break
    return model

def val_drp(val_loader, model, args):
    model.eval()
    y_true, y_pred = [], []

    logger.info("Validating...")
    for drug, cell, label in tqdm(val_loader):
        if isinstance(cell, list):
            drug, cell, label = drug.to(args.device), [feat.to(args.device) for feat in cell], label.to(args.device)
        else:
            drug, cell, label = drug.to(args.device), cell.to(args.device), label.to(args.device)
        pred = model(drug, cell)
        y_true.append(label.view(-1, 1).detach().cpu())
        y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy().flatten()
    y_pred = torch.cat(y_pred, dim=0).numpy().flatten()
    
    if args.task == "classification":
        results = {
            "roc_auc": roc_auc(y_true, y_pred)
        }
    elif args.task == "regression":
        results = {
            "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "pearson": pearsonr(y_true, y_pred)[0]
        }
    logger.info(" ".join(["%s: %.4lf" % (key, results[key]) for key in results]))
    return results

def main(args, config):
    # set random seed
    seed_all(args.seed)

    # build dataset
    dataset = SUPPORTED_DRP_DATASET[args.dataset](args.dataset_path, config["data"], task=args.task)
    collate_fn = DRPCollator(config["data"])

    # build model
    model = SUPPORTED_DRP_MODEL[config["model"]](config["network"])
    if config["model"] in ["TGSA", "TGDRP"]:
        model.cluster_predefine = {i: j.to(args.device) for i, j in dataset.predefined_cluster.items()}
        model._build()
    model = model.to(args.device)
    if args.init_checkpoint != '':
        ckpt = torch.load(args.weight_path)
        if args.param_key != '':
            ckpt = ckpt[args.param_key]
        model.load_state_dict(ckpt)

    if args.mode == "train":
        train_dataset = dataset.index_select(dataset.train_indexes)
        val_dataset = dataset.index_select(dataset.val_indexes)
        test_dataset = dataset.index_select(dataset.test_indexes)
        logger.info("# Samples: Train - %d, Val - %d, Test - %d" % (len(train_dataset), len(val_dataset), len(test_dataset)))
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        model = train_drp(train_loader, val_loader, model, args)
        val_drp(test_loader, model, args)
    elif args.mode == "test":
        val_dataset = dataset.index_select(dataset.val_indexes)
        test_dataset = dataset.index_select(dataset.test_indexes)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        model.load_state_dict(torch.load(args.weight_path, map_location=args.device)['model_state_dict'])
        val_drp(val_loader, model, args)
        val_drp(test_loader, model, args)
    elif args.mode == "zero_shot_transfer":
        nfolds = config["data"]["split"]["nfolds"]
        all_patients = ["BRCA_28", "CESC_24", "COAD_8", "GBM_69", "HNSC_45", "KIRC_47", "LUAD_23", "LUSC_20", "PAAD_55", "READ_8", "SARC_30", "SKCM_56", "STAD_30"]
        results = {x: [] for x in all_patients}
        for fold in range(nfolds):
            train_fold = list(range(fold)) + list(range(fold + 1, nfolds))
            train_dataset = dataset.index_select(np.concatenate([dataset.fold_indexes[i] for i in train_fold], axis=0))
            val_dataset = dataset.index_select(dataset.fold_indexes[fold])
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
            
            fold_model = copy.deepcopy(model)
            fold_model = train_drp(train_loader, val_loader, fold_model, args)
            val_drp(val_loader, model, args)

            for patient in all_patients:
                test_dataset = TCGA(args.transfer_dataset_path, config["data"], subset=patient)
                test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
                results[patient].append(val_drp(test_loader, model, args))
        for patient in all_patients:
            mean, std = metrics_average(results[patient])["roc_auc"]
            print("roc_auc on TCGA-%s: %.4lf±%.4lf" % (patient, mean, std))

def add_arguments(parser):
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--task', type=str, default='regression', help='task type: classification or regression')
    parser.add_argument('--dataset', type=str, default='GDSC', help='dataset')
    parser.add_argument("--dataset_path", type=str, default='../datasets/drp/GDSC/', help='path to the dataset')
    parser.add_argument('--config_path', type=str, help='path to the configuration file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers (default: 4)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10, help='patience for earlystopping (default: 10)')
    parser.add_argument('--setup', type=str, default='known', help='experimental setup')
    parser.add_argument('--init_checkpoint', type=str, default='', help='filepath for pretrained weights')
    parser.add_argument('--param_key', type=str, default='', help='the key to obtain model state dict')
    parser.add_argument('--mode', type=str, default='test', help='train, test or zero-shot transfer')

    # arguments for zero-shot transfer
    parser.add_argument("--transfer_dataset_path", type=str, default='../datasets/drp/tcga', help='path to the transfer dataset')

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path, "r"))

    main(args, config)