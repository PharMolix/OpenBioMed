import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from datasets.dp_dataset import SUPPORTED_DP_DATASETS, Task
# from models.drug_encoder import SUPPORTED_DRUG_ENCODER
from utils import DPCollator, roc_auc, EarlyStopping, AverageMeter, ToDevice
from models.dp_model import DPModel, DeepEIK4DP

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='MoleculeNet')
    parser.add_argument("--dataset_path", type=str,
                        default='../datasets/dp/MoleculeNet/')
    parser.add_argument("--dataset_name", type=str, default='BBBP')
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--output_path", type=str,
                        default="../ckpts/finetune_ckpts/dp")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--classfy_type", type=str, default="multi")
    parser.add_argument("--freeze", action="store_true")
    return parser


def get_num_task(dataset):
    dataset = dataset.lower()
    print("debug dataset:", dataset, len(dataset))
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor']:
        return 1
    elif dataset == 'pcba':
        return 92
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    raise ValueError('Invalid dataset name.')


def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index,
                         batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0
            roc_list.append(roc_auc_score(
                (y_true[is_valid, i] + 1)/2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)  # y_true.shape[1]


def get_metric(task, name):
    if task == Task.CLASSFICATION:
        metric_name = "roc_auc"
        metric = roc_auc
    elif task == Task.REGRESSION:
        if name in ["qm7", "qm8", "qm9"]:
            metric_name = "MAE"
            metric = mean_absolute_error
        else:
            metric_name = "MSE"
            metric = mean_squared_error
    return metric_name, metric


def train_dp(train_loader, val_loader, test_loader, model, task, args):
    device = torch.device(args.device)
    """
    if task == Task.CLASSFICATION:
        loss_fn = nn.CrossEntropyLoss()
        mode = "higher"
    elif task == Task.REGRESSION:
        if args.dataset_name in ["qm7", "qm8", "qm9"]:
            loss_fn = nn.L1Loss()
            mode = "lower"
        else:
            loss_fn = nn.MSELoss()
            mode = "lower"
    """
    # reduction = "none" A loss is calculated for each position, and then a total value is calculated according to the mask
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    mode = "higher"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpts_filename = args.output_path + f"/{model.name}_{model.model_num}_{args.dataset_name}" + "_finetune.pth"
    stopper = EarlyStopping(
        mode=mode, patience=args.patience, filename=ckpts_filename)
    metric_name, _ = get_metric(task, args.dataset_name)
    running_loss = AverageMeter()
    loss_list = []

    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        model.train()
        running_loss.reset()

        for step, (drug, label) in enumerate(train_loader):
            drug = ToDevice(drug, device)
            pred = model(drug)
            y = label.view(pred.shape).to(torch.float64).to(device)
            is_valid = y**2 > 0
            # Loss matrix
            loss_mat = loss_fn(pred.double(), (y+1)/2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(
                loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()

            optimizer.step()

            running_loss.update(loss.detach().cpu().item())
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" %
                            (step, running_loss.get_average()))
                loss_list.append(round(running_loss.get_average(), 3))
                running_loss.reset()
                logger.info("%s train loss = %.4f lr = %.4f" %
                    (args.dataset_name, loss_list[-1], optimizer.param_groups[0]["lr"]))
        val_metrics = val_dp(val_loader, model, task, args)
        test_metrics = val_dp(test_loader, model, task, args)
       
        logger.info("%s val %s=%.4lf loss = %.4f" %
                    (args.dataset_name, metric_name, val_metrics[metric_name], float(val_metrics["loss"])))
        logger.info("%s test %s=%.4lf loss = %.4f" % (args.dataset_name, 
                    metric_name, test_metrics[metric_name], float(test_metrics["loss"])))
        if stopper.step((val_metrics[metric_name]), model, epoch):
            break
    model.load_state_dict(torch.load(ckpts_filename)["model_state_dict"])
    return model, stopper.best_epoch, loss_list


def val_dp(val_loader, model, task, args):
    device = torch.device(args.device)
    metric_name, metric = get_metric(task, args.dataset_name)
    model.eval()
    all_preds, y_true = [], []
    for drug, label in val_loader:
        drug = ToDevice(drug, device)
        pred = model(drug).detach().cpu()
        label = label.view(pred.shape)

        all_preds.append(pred)
        y_true.append(label)
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    
    # add loss
    is_valid = torch.tensor(y_true**2 > 0)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss_mat = loss_fn(torch.tensor(all_preds).double(), torch.tensor((y_true+1)/2))
    # loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(
        loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat)/torch.sum(is_valid)

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0
            roc_list.append(roc_auc_score(
                (y_true[is_valid, i] + 1)/2, all_preds[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list))/y_true.shape[1]))

    return {metric_name: sum(roc_list)/len(roc_list), "loss": loss}  # y_true.shape[1]

    # return {metric_name: metric(all_y, all_preds)}


def main(args, config):
    # prepare dataset

    dataset = SUPPORTED_DP_DATASETS[args.dataset](
        args.dataset_path, config["data"], args.dataset_name)
    task = dataset.task
    # TODO: remove?
    targets = dataset.targets
    # normalizer = dataset.normalizer
    
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    test_dataset = dataset.index_select(dataset.test_index)
    print(train_dataset[0])

    collator = DPCollator(config["data"]["drug"])
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collator)

    # prepare model
    task_num = get_num_task(args.dataset_name)
    if len(config["data"]["drug"]["modality"]) > 1 and config["model"] not in ["molalbef", "biomedgpt"]:
        model = DeepEIK4DP(config["network"], task_num)
    else:
        model = DPModel(config, task_num)
    if args.init_checkpoint != "":
        ckpt = torch.load(args.init_checkpoint)
        if args.param_key != "" and args.param_key != "None":
            ckpt = ckpt[args.param_key]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
    
    # TODO: 冻结参数
    print("[debug] args.freeze: ", args.freeze)
    if args.freeze:
        print("====冻结参数====")
        if config["network"]["structure"] in ["graphmvp"]:
            freeze_layers = ["encoder"]
            for name, param in model.encoder.named_parameters():
                if any(nd in name for nd in freeze_layers):
                    param.requires_grad = False
        elif config["model"] in ["DeepEIK", "biomedgpt"]:
            # freeze_layers = ["drug_structure_encoder", "text_encoder"]
            freeze_layers = ["text_encoder"]
            for name, param in model.named_parameters():
                if any(nd in name for nd in freeze_layers):
                    param.requires_grad = False
        else:
            freeze_layers = ["text_encoder", "kg_encoder", "text_proj_head", "graph_encoder"]
            for name, param in model.encoder.named_parameters():
                if any(nd in name for nd in freeze_layers):
                    param.requires_grad = False
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # configure metric
    metric_name, _ = get_metric(task, args.dataset_name)

    # TODO: support two and multiple classification
    # if args.classfy_type == "binar" or "multi":
    if args.mode == "train":
        model, epoch, loss_list = train_dp(train_loader, val_loader,
                                           test_loader, model, task, args)
        results = val_dp(test_loader, model, task, args)
        logger.info("%s test %s=%.4lf" %
                    (args.dataset_name, metric_name, results[metric_name]))
    elif args.mode == "test":
        results = val_dp(test_loader, model, task, args)
        logger.info("%s test %s=%.4lf" %
                    (args.dataset_name, metric_name, results[metric_name]))

    try:
        init_ckpts = config['network']['structure']['init_checkpoint']
    except:
        init_ckpts = None
        
    module_num = len(config["data"]["drug"]["modality"])
    with open(f"{model.name}_{module_num}_{args.dataset_name}_test.txt", "a") as f:
        f.write(f"{args.dataset_name} test {metric_name}={results[metric_name]} seed {args.seed} best_epoch {epoch} batch_size {args.batch_size} lr {args.lr} weight_decay {args.weight_decay} dropout {config['network']['structure']['drop_ratio']} ckpts {init_ckpts}\n ")
        # f.write(f"test {metric_name}={results[metric_name]} best_epoch {epoch} \n ")


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
    if args.dropout:
        print("====change dropout====")
        config['network']['structure']['drop_ratio'] = args.dropout

    # set seed
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args, config)