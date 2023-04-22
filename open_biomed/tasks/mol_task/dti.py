import logging
logger = logging.getLogger(__name__)

import os
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

from datasets.dti_dataset import SUPPORTED_DTI_DATASETS
from models.dti_model import DTIModel, DeepEIK4DTI
from utils import DTICollator, AverageMeter, EarlyStopping, ToDevice, metrics_average
from utils.metrics import roc_auc, pr_auc, concordance_index, rm2_index
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def train_dti(train_loader, val_loader, model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.task == "classification":
        loss_fn = nn.CrossEntropyLoss()
        stop_mode = 'higher'
        key = "ROC_AUC"
    elif args.task == "regression":
        loss_fn = nn.MSELoss()
        stop_mode = 'lower'
        key = "MSE"
    
    running_loss = AverageMeter()
    stopper = EarlyStopping(mode=stop_mode, patience=args.patience, filename=args.output_path)
    for epoch in range(args.epochs):
        logger.info("Epoch %d" % (epoch))
        logger.info("Training...")
        model.train()
        step = 0
        for mol, prot, label in train_loader:
            mol = ToDevice(mol, device)
            prot = ToDevice(prot, device)
            label = label.to(device)
            pred = model(mol, prot)
            loss = loss_fn(pred, label)
            loss.backward()
            
            running_loss.update(loss.item(), label.size(0))
            step += 1
            #if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()

        if args.eval_train_epochs > 0 and epoch % args.eval_train_epochs == 0:
            eval_dti("train", train_loader, model, args, device)
        if val_loader is not None:
            results = eval_dti("valid", val_loader, model, args, device)
            logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))
            if stopper.step(results[key], model):
                break
    if val_loader is not None:
        model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def eval_dti(split, val_loader, model, args, device):
    model.eval()
    logger.info("Validating...")
    if args.task == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif args.task == "regression":
        loss_fn = nn.MSELoss()

    all_loss = 0
    all_preds = []
    all_labels = []
    for mol, prot, label in val_loader:
        mol = ToDevice(mol, device)
        prot = ToDevice(prot, device)
        label = label.to(device)
        pred = model(mol, prot)
        if args.task == "classification" and len(pred.shape) < 2:
            pred = pred.unsqueeze(0)
        all_loss += loss_fn(pred, label).item()
        if args.task == "classification":
            pred = F.softmax(pred, dim=-1)[:, 1]
        
        all_preds.append(np.array(pred.detach().cpu()))
        all_labels.append(np.array(label.detach().cpu()))
    logger.info("Average %s loss: %.4lf" % (split, all_loss / len(val_loader)))
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if args.task == "classification":
        outputs = np.array([1 if x >= 0.5 else 0 for x in all_preds])
        results = {
            "ROC_AUC": roc_auc(all_labels, all_preds), 
            "PR_AUC": pr_auc(all_labels, all_preds), 
            "F1": f1_score(all_labels, outputs),
            "Precision": precision_score(all_labels, outputs),
            "Recall": recall_score(all_labels, outputs), 
        }
    elif args.task == "regression":
        results = {
            "MSE": mean_squared_error(all_labels, all_preds),
            "Pearson": pearsonr(all_labels, all_preds)[0],
            "Spearman": spearmanr(all_labels, all_preds)[0],
            "CI": concordance_index(all_labels, all_preds),
            "r_m^2": rm2_index(all_labels, all_preds)
        }
    return results

def main(args, config):
    # configure seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare dataset
    if args.dataset in ['yamanishi08', 'bmkg-dti']:
        args.task = "classification"
        pred_dim = 2
    else:
        args.task = "regression"
        pred_dim = 1
    dataset = SUPPORTED_DTI_DATASETS[args.dataset](args.dataset_path, config["data"], args.split_strategy)

    if args.mode == "train":
        train_dataset = dataset.index_select(dataset.train_index)
        if len(dataset.val_index) > 0:
            val_dataset = dataset.index_select(dataset.val_index)
        else:
            val_dataset = None
        test_dataset = dataset.index_select(dataset.test_index)
        train_dataset._build(
            test_dataset.pair_index, 
            None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(config["data"]["drug"]["featurizer"]["kg"]["save_path"], "dti-" + args.dataset + ".pkl")
        )
        if val_dataset is not None:
            val_dataset._build(
                test_dataset.pair_index, 
                None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(config["data"]["drug"]["featurizer"]["kg"]["save_path"], "dti-" + args.dataset + ".pkl")
            )
        test_dataset._build(
            test_dataset.pair_index, 
            None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(config["data"]["drug"]["featurizer"]["kg"]["save_path"], "dti-" + args.dataset + ".pkl")
        )
        collator = DTICollator(config["data"])
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
        else:
            val_loader = None
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
        
        if len(config["data"]["drug"]["modality"]) > 1:
            model = DeepEIK4DTI(config["network"], pred_dim)
        else:
            model = DTIModel(config["network"], pred_dim)
        if args.init_checkpoint != "None":
            ckpt = torch.load(args.init_checkpoint)
            if args.param_key != "None":
                ckpt = ckpt[args.param_key]
            model.load_state_dict(ckpt)
        device = torch.device(args.device)
        model = model.to(device)
        
        model = train_dti(train_loader, test_loader, model, args, device)
        results = eval_dti("test", test_loader, model, args, device)
        logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))
    elif args.mode == "kfold":
        results = []
        for i in range(dataset.nfolds):
            logger.info("Fold %d", i)
            train_dataset = dataset.index_select(dataset.folds[i]["train"])
            test_dataset = dataset.index_select(dataset.folds[i]["test"])
            train_dataset._build(
                test_dataset.pair_index, 
                None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(config["data"]["drug"]["featurizer"]["kg"]["save_path"], "dti-" + args.dataset + "-" + args.split_strategy + "-fold" + str(i) + ".pkl")
            )
            test_dataset._build(
                test_dataset.pair_index, 
                None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(config["data"]["drug"]["featurizer"]["kg"]["save_path"], "dti-" + args.dataset + "-" + args.split_strategy + "-fold" + str(i) + ".pkl")
            )
            collator = DTICollator(config["data"])
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
            
            # prepare model
            if len(config["data"]["drug"]["modality"]) > 1:
                model = DeepEIK4DTI(config["network"], pred_dim)
            else:
                model = DTIModel(config["network"], pred_dim)
            #print(model)
            if args.init_checkpoint != "None":
                ckpt = torch.load(args.init_checkpoint)
                if args.param_key != "None":
                    ckpt = ckpt[args.param_key]
                model.load_state_dict(ckpt)
            device = torch.device(args.device)
            model = model.to(device)
            
            model = train_dti(train_loader, test_loader, model, args, device)
            results.append(eval_dti("test", test_loader, model, args, device))
        results = metrics_average(results)
        for key in results:
            print("%s: %.4lf±%.4lf" % (key, results[key][0], results[key][1]))

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default="Yamanishi08")
    parser.add_argument("--dataset_path", type=str, default="../datasets/dti/Yamanishi08/")
    parser.add_argument("--split_strategy", type=str, default="random")
    parser.add_argument("--init_checkpoint", type=str, default="None")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/dp/finetune.pth")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_train_epochs", type=int, default=-1)
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
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = json.load(open(args.config_path, "r"))
    main(args, config)
    
    