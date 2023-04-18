import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import math
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import EarlyStopping, AverageMeter, DrugCollator, ToDevice, recall_at_k
from utils.optimizers import BertAdam
from datasets.mtr_dataset import SUPPORTED_MTR_DATASETS
from models.drug_encoder import KVPLM, MoMu, DrugBERT, BioMedGPT
from models.mtr_model import MTRModel

SUPPORTED_MTR_MODEL = {
    "SciBERT": DrugBERT,
    "KV-PLM": KVPLM, 
    "KV-PLM*": KVPLM,
    "MoMu": MoMu, 
    "BioMedGPT": BioMedGPT,
    "combined": MTRModel
}

def contrastive_loss(logits_des, logits_smi, margin, device):
    scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
    diagonal = scores.diag().view(logits_smi.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_des = (margin + scores - d1).clamp(min=0)
    cost_smi = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.to(device)
    cost_des = cost_des.masked_fill_(I, 0)
    cost_smi = cost_smi.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    #if self.max_violation:
    cost_des = cost_des.max(1)[0]
    cost_smi = cost_smi.max(0)[0]

    return cost_des.sum() + cost_smi.sum()

def train_mtr(train_dataset, val_dataset, model, collator, args):
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
    loss_fn = contrastive_loss
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },{
        'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        weight_decay=0,
        lr=args.lr,
        warmup=args.warmup,
        t_total=len(train_loader) * args.epochs,
    )
    stopper = EarlyStopping(mode="higher", patience=args.patience, filename=args.output_path)

    running_loss = AverageMeter()
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        model.train()
        running_loss.reset()

        step = 0
        for drug in tqdm(train_loader):
            drug = ToDevice(drug, args.device)
            drug_rep = model.encode_structure(drug["structure"])
            text_rep = model.encode_text(drug["text"])
            loss = loss_fn(drug_rep, text_rep, margin=args.margin, device=args.device)
            if hasattr(model, "calculate_matching_loss"):
                matching_loss = model.calculate_matching_loss(drug["structure"], drug["text"])
                loss += matching_loss
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.log_every == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()

        val_metrics = val_mtr(val_dataset, model, collator, args)
        logger.info(", ".join(["val %s: %.4lf" % (k, val_metrics[k]) for k in val_metrics]))
        if stopper.step((val_metrics["mrr_d2t"] + val_metrics["mrr_t2d"]), model):
            break
    model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def rerank(dataset, model, index_structure, index_text, score, alpha, collator, device):
    mini_batch = []
    for i in index_structure:
        for j in index_text:
            mini_batch.append({
                "structure": dataset[i]["structure"],
                "text": dataset[j]["text"],
            })
    mini_batch = ToDevice(collator(mini_batch), device)
    score = score.to(device) * alpha + model.predict_similarity_score(mini_batch).squeeze() * (1 - alpha)
    _, new_idx = torch.sort(score, descending=True)
    if len(index_structure) > 1:
        return torch.LongTensor([index_structure[i] for i in new_idx.detach().cpu().tolist()])
    else:
        return torch.LongTensor([index_text[i] for i in new_idx.detach().cpu().tolist()])

def val_mtr(val_dataset, model, collator, args):
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    model.eval()
    drug_rep_total, text_rep_total = [], []
    n_samples = 0
    with torch.no_grad():
        for drug in tqdm(val_loader):
            drug = ToDevice(drug, args.device)

            drug_rep = model.encode_structure(drug["structure"])
            text_rep = model.encode_text(drug["text"])
            drug_rep_total.append(drug_rep)
            text_rep_total.append(text_rep)
            
            # calculate #1 acc
            """
            scores = torch.cosine_similarity(drug_rep.unsqueeze(1).expand(drug_rep.shape[0], drug_rep.shape[0], drug_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            mx_d2t = torch.argmax(scores, axis=1)
            mx_t2d = torch.argmax(scores, axis=0)
            acc_d2t += sum((mx_d2t == torch.arange(mx_d2t.shape[0]).to(args.device)).int()).item()
            acc_t2d += sum((mx_t2d == torch.arange(mx_t2d.shape[0]).to(args.device)).int()).item()
            """
            n_samples += drug_rep.shape[0]

        drug_rep = torch.cat(drug_rep_total, dim=0)
        text_rep = torch.cat(text_rep_total, dim=0)
        score = torch.zeros(n_samples, n_samples)
        mrr_m2t, mrr_t2m = 0, 0
        rec_m2t, rec_t2m = [0, 0, 0], [0, 0, 0]
        logger.info("Calculating cosine similarity...")
        for i in tqdm(range(n_samples)):
            score[i] = torch.cosine_similarity(drug_rep[i], text_rep)
        if hasattr(model, "predict_similarity_score") and args.rerank:
            logger.info("Reranking...")
        for i in tqdm(range(n_samples)):
            _, idx = torch.sort(score[i, :], descending=True)
            idx = idx.detach().cpu()
            if hasattr(model, "predict_similarity_score") and args.rerank:
                idx = torch.cat((
                    rerank(val_dataset, model, [i], idx[:args.rerank_num].tolist(), score[i, idx[:args.rerank_num]], args.alpha_m2t, collator, args.device),
                    idx[args.rerank_num:]
                ), dim=0)
            for j, k in enumerate([1, 5, 10]):
                rec_m2t[j] += recall_at_k(idx, i, k)
            mrr_m2t += 1.0 / ((idx == i).nonzero(as_tuple=True)[0].item() + 1)

            _, idx = torch.sort(score[:, i], descending=True)
            idx = idx.detach().cpu()
            if hasattr(model, "predict_similarity_score") and args.rerank:
                idx = torch.cat((
                    rerank(val_dataset, model, idx[:args.rerank_num].tolist(), [i], score[idx[:args.rerank_num], i], args.alpha_t2m, collator, args.device),
                    idx[args.rerank_num:]
                ), dim=0)
            for j, k in enumerate([1, 5, 10]):
                rec_t2m[j] += recall_at_k(idx, i, k)
            mrr_t2m += 1.0 / ((idx == i).nonzero(as_tuple=True)[0].item() + 1)

        result = {
            "mrr_d2t": mrr_m2t / n_samples,
            "mrr_t2d": mrr_t2m / n_samples,
        }
        for idx, k in enumerate([1, 5, 10]):
            result["rec@%d_d2t" % k] = rec_m2t[idx] / n_samples
            result["rec@%d_t2d" % k] = rec_t2m[idx] / n_samples
        return result

def main(args, config):
    dataset = SUPPORTED_MTR_DATASETS[args.dataset](args.dataset_path, config["data"], args.dataset_mode, args.filter, args.filter_path)
    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    test_dataset = dataset.index_select(dataset.test_index)
    val_dataset.set_test()
    test_dataset.set_test()
    collator = DrugCollator(config["data"]["drug"])

    model = SUPPORTED_MTR_MODEL[config["model"]](config["network"])
    if args.init_checkpoint != "None":
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if args.param_key != "None":
            ckpt = ckpt[args.param_key]
        model.load_state_dict(ckpt)
    model = model.to(args.device)
    
    if args.mode == "zero_shot":
        model.eval()
        result = val_mtr(test_dataset, model, collator, args)
        print(result)
    elif args.mode == "train":
        train_mtr(train_dataset, val_dataset, model, collator, args)
        result = val_mtr(test_dataset, model, collator, args)
        print(result)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="zero_shot")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='PCdes')
    parser.add_argument("--dataset_path", type=str, default='../datasets/mtr/PCdes/')
    parser.add_argument("--dataset_mode", type=str,  default="paragraph")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--filter_path", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="None")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/finetune.pth")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.2)
    rerank_group = parser.add_mutually_exclusive_group(required=False)
    rerank_group.add_argument("--rerank", action="store_true")
    rerank_group.add_argument("--no_rerank", action="store_false")
    parser.set_defaults(rerank=False)
    parser.add_argument("--rerank_num", type=int, default=32)
    parser.add_argument("--alpha_m2t", type=float, default=0.8)
    parser.add_argument("--alpha_t2m", type=float, default=0.8)

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
    if args.dataset_mode == "sentence":
        config["data"]["drug"]["featurizer"]["text"]["name"] = "TransformerSentenceTokenizer"
        config["data"]["drug"]["featurizer"]["text"]["min_sentence_length"] = 5

    main(args, config)