import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import json
import math
from tqdm import tqdm
from transformers import BertTokenizer
import pickle

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from open_biomed.utils import EarlyStopping, AverageMeter, MTCollator, ToDevice, recall_at_k
from open_biomed.utils.optimizers import BertAdam
from open_biomed.datasets.mtr_dataset import SUPPORTED_MTR_DATASETS
from open_biomed.models.multimodal import KVPLM, MolBERT, BioMedGPTCLIP, MoMu, MolFM, DrugFM, MoleculeSTM, MolKFormer, MolCAStage1, TDMoLMStage1, CLAMP, MVMol
from open_biomed.models.task_model.mtr_model import MTRModel

SUPPORTED_MTR_MODEL = {
    "scibert": MolBERT,
    "kv-plm": KVPLM, 
    "kv-plm*": KVPLM,
    "momu": MoMu, 
    "molfm": MolFM,
    "drugfm": DrugFM,
    "biomedgpt": BioMedGPTCLIP,
    "molkformer": MolKFormer,
    "moleculestm": MoleculeSTM,
    "clamp": CLAMP,
    "molca": MolCAStage1,
    "3d-molm": TDMoLMStage1,
    "mvmol": MVMol,
    "combined": MTRModel
}

def mtr_encode_mol(model, mol, view_oper="add"):
    if isinstance(mol, dict) and "text" in mol and view_oper != "hybrid":
        mol_rep = model.encode_mol(mol)
        #if hasattr(model, "structure_proj_head"):
        #    mol_rep = model.structure_proj_head(mol_rep)
        if view_oper == "add":
            text_rep = model.encode_text(mol["text"])
            if hasattr(model, "text_proj_head"):
                text_rep = model.text_proj_head(text_rep)
            mol_rep = mol_rep + text_rep
    else:
        mol_rep = model.encode_mol(mol)
        if hasattr(model, "structure_proj_head"):
            mol_rep = model.structure_proj_head(mol_rep)
    if model.norm:
        mol_rep = F.normalize(mol_rep, dim=-1)
    return mol_rep

def mtr_encode_text(model, text):
    text_rep = model.encode_text(text)
    #if hasattr(model, "text_proj_head"):
    #    text_rep = model.text_proj_head(text_rep)
    if model.norm:
        text_rep = F.normalize(text_rep, dim=-1)
    return text_rep

def similarity(logits1, logits2):
    if len(logits1.shape) >= 2: 
        sim = logits1 @ logits2.transpose(0, 1)
        sim, _ = torch.max(sim, dim=0)
    else:
        sim = torch.cosine_similarity(logits1, logits2)
        
    return sim

def contrastive_loss(logits_structure, logits_text, margin, device):
    if len(logits_structure.shape) <= 2:
        scores = torch.cosine_similarity(
            logits_structure.unsqueeze(1).expand(logits_structure.shape[0], logits_structure.shape[0], logits_structure.shape[1]), 
            logits_text.unsqueeze(0).expand(logits_text.shape[0], logits_text.shape[0], logits_text.shape[1]), 
            dim=-1
        )
    else:
        scores = torch.matmul(logits_structure.unsqueeze(1), logits_text.unsqueeze(-1)).squeeze()
        scores, _ = scores.max(dim=-1) 
    diagonal = scores.diag().view(logits_text.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_s2t = (margin + scores - d1).clamp(min=0)
    cost_t2s = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.to(device)
    cost_s2t = cost_s2t.masked_fill_(I, 0)
    cost_t2s = cost_t2s.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    #if self.max_violation:
    cost_s2t = cost_s2t.max(1)[0]
    cost_t2s = cost_t2s.max(0)[0]

    return cost_s2t.sum() + cost_t2s.sum()


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
        for mol, text in tqdm(train_loader):
            mol = ToDevice(mol, args.device)
            text = ToDevice(text, args.device)
            mol_rep = mtr_encode_mol(model, mol, args.view_operation)
            text_rep = mtr_encode_text(model, text)
            loss = loss_fn(mol_rep, text_rep, margin=args.margin, device=args.device)
            #if hasattr(model, "calculate_matching_loss"):
            #    matching_loss = model.calculate_matching_loss(mol, text)
            #    loss += matching_loss
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.log_every == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()

        val_metrics = val_mtr(val_dataset, model, collator, False, args)
        logger.info(", ".join(["val %s: %.4lf" % (k, val_metrics[k]) for k in val_metrics]))
        if stopper.step((val_metrics["mrr_d2t"] + val_metrics["mrr_t2d"]), model):
            break
    model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def rerank(dataset, model, index_structure, index_text, score, alpha, collator, device):
    mini_batch_mol, mini_batch_text = [], []
    for i in index_structure:
        for j in index_text:
            mini_batch_mol.append(dataset[i][0])
            mini_batch_text.append(dataset[j][1])

    mini_batch_mol = ToDevice(collator.mol_collator(mini_batch_mol), device)
    mini_batch_text = ToDevice(collator.text_collator(mini_batch_text), device)
    #print(index_structure, index_text, score, model.predict_similarity_score(mini_batch).squeeze())
    score = score.to(device) * alpha + model.predict_similarity_score(mini_batch_mol, mini_batch_text).squeeze() * (1 - alpha)
    _, new_idx = torch.sort(score, descending=True)
    if len(index_structure) > 1:
        return torch.LongTensor([index_structure[i] for i in new_idx.detach().cpu().tolist()])
    else:
        return torch.LongTensor([index_text[i] for i in new_idx.detach().cpu().tolist()])

def val_mtr(val_dataset, model, collator, apply_rerank, args):
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    model.eval()
    mol_rep_total, text_rep_total = [], []
    mol_rep_noview, views = [], []
    tokenizer = BertTokenizer.from_pretrained("./ckpts/text_ckpts/scibert_scivocab_uncased")
    n_samples = 0
    with torch.no_grad():
        for mol, text in tqdm(val_loader):
            if "text" in mol:
                for view in tokenizer.batch_decode(mol["text"]["input_ids"]):
                    views.append(view)

            mol = ToDevice(mol, args.device)
            text = ToDevice(text, args.device)
            
            #print(mol)
            #print(text)
            mol_rep = mtr_encode_mol(model, mol, args.view_operation)
            text_rep = mtr_encode_text(model, text)
            # mol.pop("text")
            # mol_rep_noview.append(mtr_encode_mol(model, mol, args.view_operation))
            mol_rep_total.append(mol_rep)
            text_rep_total.append(text_rep)
            
            n_samples += mol_rep.shape[0]

        mol_rep = torch.cat(mol_rep_total, dim=0)
        text_rep = torch.cat(text_rep_total, dim=0)
        pickle.dump({
            # "mol_rep_noview": torch.cat(mol_rep_noview, dim=0).cpu(),
            "mol_rep": mol_rep.cpu(),
            "text_rep": text_rep.cpu(),
            "view": views
        }, open("./assets/moleculestm_intermediate.pkl", "wb"))
        score = torch.zeros(n_samples, n_samples)
        mrr_m2t, mrr_t2m = 0, 0
        rec_m2t, rec_t2m = [0, 0, 0], [0, 0, 0]
        logger.info("Calculating cosine similarity...")
        for i in range(n_samples):
            score[i] = similarity(mol_rep[i], text_rep)
        if hasattr(model, "predict_similarity_score") and apply_rerank:
            logger.info("Reranking...")
        for i in tqdm(range(n_samples)):
            _, idx = torch.sort(score[i, :], descending=True)
            idx = idx.detach().cpu()
            if hasattr(model, "predict_similarity_score") and apply_rerank and recall_at_k(idx, i, args.rerank_num):
                idx = torch.cat((
                    rerank(val_dataset, model, [i], idx[:args.rerank_num].tolist(), score[i, idx[:args.rerank_num]], args.alpha_m2t, collator, args.device),
                    idx[args.rerank_num:]
                ), dim=0)
            for j, k in enumerate([1, 5, 10]):
                rec_m2t[j] += recall_at_k(idx, i, k)
            mrr_m2t += 1.0 / ((idx == i).nonzero(as_tuple=True)[0].item() + 1)

            _, idx = torch.sort(score[:, i], descending=True)
            idx = idx.detach().cpu()
            if hasattr(model, "predict_similarity_score") and apply_rerank and recall_at_k(idx, i, args.rerank_num):
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
    dataset = SUPPORTED_MTR_DATASETS[args.dataset](args.dataset_path, config["data"], args.dataset_mode, args.perspective, args.filter, args.filter_path)
    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    test_dataset = dataset.index_select(dataset.test_index)
    val_dataset.set_test()
    test_dataset.set_test()
    collator = MTCollator(config["data"])

    model = SUPPORTED_MTR_MODEL[config["model"]](config["network"])
    if args.init_checkpoint != "None":
        logger.info("Load checkpoint from %s" % (args.init_checkpoint))
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if args.param_key != "None":
            ckpt = ckpt[args.param_key]
        model.load_state_dict(ckpt, strict=False)
    model = model.to(args.device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.mode == "zero_shot":
        model.eval()
        result = val_mtr(test_dataset, model, collator, args.rerank, args)
        print(result)
    elif args.mode == "train":
        train_mtr(train_dataset, val_dataset, model, collator, args)
        result = val_mtr(test_dataset, model, collator, args.rerank, args)
        print(result)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="zero_shot")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='PCdes')
    parser.add_argument("--dataset_path", type=str, default='../datasets/mtr/PCdes/')
    parser.add_argument("--dataset_mode", type=str, default="paragraph")
    parser.add_argument("--view_operation", type=str, default="add")
    parser.add_argument("--perspective", type=str, default="None")
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
        config["data"]["mol"]["featurizer"]["text"]["name"] = "TransformerSentenceTokenizer"
        config["data"]["mol"]["featurizer"]["text"]["min_sentence_length"] = 5

    main(args, config)