import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import random
import argparse
import json
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from open_biomed.datasets.ddi_dataset import SUPPORTED_DDI_DATASETS, DrugBank
from open_biomed.models.task_model.ddi_model import SUPPORTED_DDI_NETWORKS
from open_biomed.utils import DDICollator, AverageMeter, EarlyStopping, ToDevice, metrics_average
from open_biomed.utils.metrics import roc_auc, pr_auc, concordance_index, rm2_index
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr


def train_ddi_net(train_loader, val_loader, model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    stop_mode = 'higher'
    key = "F1"

    running_loss = AverageMeter()
    stopper = EarlyStopping(mode=stop_mode, patience=args.patience, filename=args.output_path)
    for epoch in range(args.epochs):
        logger.info("Epoch %d" % (epoch))
        logger.info("Training...")
        model.train()
        step = 0
        for molA, molB, label in train_loader:
            molA = ToDevice(molA, device)
            molB = ToDevice(molB, device)
            label = label.to(device)

            pred = model(molA, molB)
            loss = loss_fn(pred, label)
            loss.backward()

            running_loss.update(loss.item(), label.size(0))
            step += 1
            # if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()

        # if args.eval_train_epochs > 0 and epoch % args.eval_train_epochs == 0:
        #     eval_ddi_net("train", train_loader, model, args, device)
        if val_loader is not None:
            results = eval_ddi_net("valid", val_loader, model, args, device)
            logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))
            if stopper.step(results[key], model, epoch):
                break

    if val_loader is not None:
        model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model


def eval_ddi_net(split, val_loader, model, args, device):
    model.mode = 'eval'
    model.eval()
    logger.info("Validating...")
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    all_loss = 0
    all_preds = []
    all_labels = []
    for molA, molB, label in val_loader:
        molA = ToDevice(molA, device)
        molB = ToDevice(molB, device)
        label = label.to(device)
        pred = model(molA, molB)
        if len(pred.shape) < 2:
            pred = pred.unsqueeze(0)
        all_loss += loss_fn(pred, label).item()
        pred = torch.argmax(pred, dim=-1)
        #print(pred, torch.argmax(label, dim=-1))

        all_preds.append(np.array(pred.detach().cpu()))
        all_labels.append(np.array(torch.argmax(label, dim=-1).detach().cpu()))
    logger.info("Average %s loss: %.4lf" % (split, all_loss / len(val_loader)))
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    """
    outputs = np.array([1 if x >= 0.5 else 0 for x in all_preds])
    results = {
        "ROC_AUC": roc_auc(all_labels, all_preds),
        "PR_AUC": pr_auc(all_labels, all_preds),
        "F1": f1_score(all_labels, outputs),
        "Precision": precision_score(all_labels, outputs),
        "Recall": recall_score(all_labels, outputs),
    }
    """
    results = {
        "F1": f1_score(all_labels, all_preds, average='micro')
    }
    return results

def train_ddi_other(train_loader, model):
    feats = []
    labels = []
    for molA, molB, label in train_loader:
        feats.append(torch.hstack((molA, molB)))
        labels.append(label)
    feats = torch.vstack(feats).numpy()
    labels = torch.hstack(labels).numpy()
    model.fit(feats, labels)

def eval_ddi_other(test_loader, model):
    feats = []
    labels = []
    for molA, molB, label in test_loader:
        feats.append(torch.hstack((molA, molB)))
        labels.append(label)
    feats = torch.vstack(feats).numpy()
    pred = model.predict(feats)
    labels = torch.hstack(labels).numpy()
    results = {
        "ROC_AUC": roc_auc(labels, pred),
        "PR_AUC": pr_auc(labels, pred),
        "F1": f1_score(labels, pred),
        "Precision": precision_score(labels, pred),
        "Recall": recall_score(labels, pred),
    }
    return results

def main(args, config):
    # configure seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pred_dim = 86
    # dataset = SUPPORTED_DDI_DATASETS[args.dataset](args.dataset_path, config["data"], args.split_strategy)
    device = torch.device(args.device)

    if args.mode == "train":
        dataset = DrugBank(args.dataset_path, config["data"]["drug"])
        train_dataset = dataset.index_select(dataset.train_index)
        val_dataset = dataset.index_select(dataset.val_index)
        test_dataset = dataset.index_select(dataset.test_index)
        print(len(train_dataset), len(val_dataset), len(test_dataset))

        if config['model'] in SUPPORTED_DDI_NETWORKS:
            collator = DDICollator(config["data"])
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      collate_fn=collator)
            if val_dataset is not None:
                val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                        collate_fn=collator)
            else:
                val_loader = None
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                     collate_fn=collator)

            model = SUPPORTED_DDI_NETWORKS[config['model']](config["network"], pred_dim)
            if args.init_checkpoint != "None":
                ckpt = torch.load(args.init_checkpoint)
                if args.param_key != "None":
                    ckpt = ckpt[args.param_key]
                model.load_state_dict(ckpt)
                logger.info("Loaded " + args.init_checkpoint)
            model = model.to(device)
            logging.info('Number of trainable parameters: ' + str(sum(p.numel() for p in model.parameters())))
            model = train_ddi_net(train_loader, val_loader, model, args, device)
            results = eval_ddi_net("test", test_loader, model, args, device)
        else:
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        logger.info(", ".join(["%s: %.4lf" % (k, v) for k, v in results.items()]))

    elif args.mode == "kfold":
        results = []
        # for i in range(dataset.nfolds):
        for i in range(10):
            logger.info("Fold %d", i)
            dataset_filename = args.dataset + '-' + args.split_strategy + '-fold' + str(i) + '.pkl'
            if dataset_filename in os.listdir(args.dataset_path):
                logger.info('Loading dataset for fold ' + str(i) + ' from file')
                with open(os.path.join(args.dataset_path, dataset_filename), 'rb') as f:
                    train_dataset, test_dataset = pickle.load(f)
                logger.info('Done')
            else:
                raise RuntimeError('Datasets should not have to be re-built.')
                logger.info('Building dataset for fold ' + str(i))
                train_dataset = dataset.index_select(dataset.folds[i]["train"])
                test_dataset = dataset.index_select(dataset.folds[i]["test"])
                train_dataset._build(
                    test_dataset.pair_index,
                    None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(
                        config["data"]["drug"]["featurizer"]["kg"]["save_path"],
                        "ddi-" + args.dataset + "-" + args.split_strategy + "-fold" + str(i) + ".pkl")
                )
                test_dataset._build(
                    test_dataset.pair_index,
                    None if "kg" not in config["data"]["drug"]["modality"] else os.path.join(
                        config["data"]["drug"]["featurizer"]["kg"]["save_path"],
                        "ddi-" + args.dataset + "-" + args.split_strategy + "-fold" + str(i) + ".pkl")
                )
                with open(os.path.join(args.dataset_path, dataset_filename), 'wb') as f:
                    pickle.dump((train_dataset, test_dataset), f)
            # continue
            all_pair_index = train_dataset.pair_index
            train_dataset, val_dataset = train_dataset.split_train_val(len(test_dataset))
            assert set(all_pair_index) == set(train_dataset.pair_index).union(val_dataset.pair_index)
            print('Proportion of train in train+val:', len(train_dataset) / (len(train_dataset) + len(val_dataset)))
            print('Proportion fo val in train+val:', len(val_dataset) / (len(train_dataset) + len(val_dataset)))

            kge = train_dataset.kge  # same as val_dataset.kge and test_dataset.kge
            if kge is not None:
                kge = {k: torch.from_numpy(kge[k]).to(device) for k in kge}
            if 'network' in config:
                config['network']['kge'] = kge

            # prepare model
            collator = DDICollator(config["data"])
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                        collate_fn=collator)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    collate_fn=collator)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                        collate_fn=collator)

            model = SUPPORTED_DDI_NETWORKS[config['model']](config["network"], pred_dim)
            if args.init_checkpoint != "None":
                ckpt = torch.load(args.init_checkpoint)
                if args.param_key != "None":
                    ckpt = ckpt[args.param_key]
                model.load_state_dict(ckpt)
            model = model.to(device)
            logging.info('Number of trainable parameters: ' + str(sum(p.numel() for p in model.parameters())))
            model = train_ddi_net(train_loader, val_loader, model, args, device)
            fold_result = eval_ddi_net("test", test_loader, model, args, device)
            print('Fold', i, 'results:')
            print(fold_result, '\n')
            results.append(fold_result)

        results = metrics_average(results)
        for key in results:
            print("%s: %.4lf±%.4lf" % (key, results[key][0], results[key][1]))


def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="kfold")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default="mssl2drug")
    parser.add_argument("--dataset_path", type=str, default="../datasets/ddi/MSSL2drug/")
    parser.add_argument("--split_strategy", type=str, default="warm")
    parser.add_argument("--init_checkpoint", type=str, default="None")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/ddi/finetune.pth")
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