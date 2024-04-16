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
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from open_biomed.datasets.moltextgen_dataset import SUPPORTED_MOLCAP_DATASET
from open_biomed.models.multimodal.text2mol import Text2MolMLP
from open_biomed.models.task_model.molcap_model import MolCapModel, GraphEnhancedMolCapModel
from open_biomed.models.multimodal import MolKFormer, MVMol, TDMoLMStage2, DrugFM

from utils.distributed_utils import init_distributed_mode, add_ddp_arguments, concat_gather, is_main_process
from utils import AverageMeter, ToDevice, MTCollator

SUPPORTED_MOLCAP_MODEL = {
    "molt5": MolCapModel,
    "chatmol": MolCapModel,
    "biot5": MolCapModel,
    "drugfm": DrugFM,
    "graph-enhanced": GraphEnhancedMolCapModel,
    "molkformer": MolKFormer,
    "3d-molm": TDMoLMStage2,
    "mvmol": MVMol,
}

def train_molcap(train_loader, val_loader, test_loader, test_dataset, model, model_without_ddp, decode_tokenizer, args, device):
    requires_grad = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            requires_grad.append(k)
    logger.debug("parameters requires grad: %s" % (" ".join(requires_grad)))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    schedular = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=args.warmup_epochs * 1.0 / args.epochs,
        anneal_strategy='cos',
        steps_per_epoch=len(train_loader),
        final_div_factor=1,
        epochs=args.epochs
    )

    running_loss = AverageMeter()
    step = 0
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        model.train()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        pre_loss = 100
        for mol, text in train_loader:
            mol = ToDevice(mol, device)
            text = ToDevice(text, device)
            loss = model_without_ddp.causal_generation_loss(mol, text)
            
            loss.backward()
            #print(nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=1.0))
            nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=0.1)
            cur_loss = loss.detach().cpu().item()
            #if cur_loss - pre_loss > 1.5:
            #    print("collapse")
            pre_loss = cur_loss
            #print(cur_loss)

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                schedular.step()
                optimizer.zero_grad()
            running_loss.update(loss.detach().cpu().item())
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf, lr=%.6lf" % (step, running_loss.get_average(), optimizer.param_groups[0]["lr"]))
                running_loss.reset()
        val_molcap(val_loader, model, model_without_ddp, device)
        if (epoch + 1) % args.eval_epochs == 0 and epoch / args.epochs > 0:
            if is_main_process():
                torch.save({'model_state_dict': model_without_ddp.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
            #print(test_molcap(val_dataset, val_loader, model, args, device, report_text2mol=False))
            print(test_molcap(test_dataset, test_loader, model, model_without_ddp, decode_tokenizer, args, device, report_text2mol=False))
        if args.distributed:
            dist.barrier()
    return model

def val_molcap(val_loader, model, model_without_ddp, device):
    model.eval()
    val_loss = 0

    logger.info("Validating...")
    with torch.no_grad():
        for mol, text in val_loader:
            mol = ToDevice(mol, device)
            text = ToDevice(text, device)
            loss = model_without_ddp.causal_generation_loss(mol, text)
            val_loss += loss.detach().cpu().item()
    logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def test_molcap(test_dataset, test_loader, model, model_without_ddp, decode_tokenizer, args, device, report_text2mol=True):
    model.eval()
    outputs = []
    gts = test_dataset.texts_raw

    logger.info("Testing...")
    if report_text2mol:
        text2mol = Text2MolMLP(
            ninp=768, 
            nhid=600, 
            nout=300, 
            model_name_or_path=args.text2mol_bert_path, 
            cid2smiles_path=os.path.join(args.text2mol_data_path, "cid_to_smiles.pkl"),
            cid2vec_path=os.path.join(args.text2mol_data_path, "test.txt")
        )
        text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path), strict=False)
        device = torch.device(args.device)
        text2mol.to(device)
    with torch.no_grad():
        for i, (mol, text) in enumerate(tqdm(test_loader)):
            mol = ToDevice(mol, device)
            output = model_without_ddp.decode(mol, num_beams=1, max_length=512)
            if i <= 3:
                decoded = decode_tokenizer.batch_decode(output, skip_special_tokens=True)
                for j in range(5):
                    logger.info("Generated: %s" % decoded[-j])
                    logger.info("Ground truth: %s" % gts[(len(outputs) + 1) * output.shape[0] - j])
                    logger.info("------------------------------------------------------")
            output = torch.cat([output, torch.ones(output.shape[0], 512 - output.shape[1]).long().to(device) * decode_tokenizer.eos_token_id], dim=1)
            outputs.append(output)
            
    outputs = torch.cat(outputs, dim=0)
    if args.distributed:
        outputs = concat_gather(outputs)

    #if not is_main_process():
    #    return {}
    outputs = decode_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.text2mol_bert_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    text2mol_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    with open(args.caption_save_path, "w", encoding='utf-8') as f:
        f.write("SMILES\tground truth\toutput\n")
        for i in range(len(outputs)):
            output_tokens.append(tokenizer.tokenize(outputs[i], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(gts[i], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
            rouge_scores.append(scorer.score(outputs[i], gts[i]))
            if report_text2mol:
                text2mol_scores.append(text2mol(test_dataset.smiles[i], outputs[i], device).detach().cpu().item())
            try:
                f.write(test_dataset.smiles[i] + '\t' + gts[i] + '\t' + outputs[i] + '\n')
            except:
                continue
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    results = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
    }
    if report_text2mol:
        results["Text2Mol"] = np.mean(text2mol_scores)
    return results

def test_molcap_from_file(file, args, device, report_text2mol=True):
    tokenizer = BertTokenizerFast.from_pretrained(args.text2mol_bert_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    text2mol_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    if report_text2mol:
        text2mol = Text2MolMLP(
            ninp=768, 
            nhid=600, 
            nout=300, 
            model_name_or_path=args.text2mol_bert_path, 
            cid2smiles_path=os.path.join(args.text2mol_data_path, "cid_to_smiles.pkl"),
            cid2vec_path=os.path.join(args.text2mol_data_path, "test.txt")
        )
        text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path), strict=False)
        device = torch.device(args.device)
        text2mol.to(device)
    with open(file, "r") as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            line = line.rstrip("\n").split("\t")
            output_tokens.append(tokenizer.tokenize(line[1], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(line[2], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
            rouge_scores.append(scorer.score(line[1], line[2]))
            if report_text2mol:
                text2mol_scores.append(text2mol(line[0], line[1], device).detach().cpu().item())
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    results = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
    }
    if report_text2mol:
        results["Text2Mol"] = np.mean(text2mol_scores)
    return results

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='chebi-20')
    parser.add_argument("--dataset_path", type=str, default='../datasets/molcap/chebi-20')
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/caption.pth")
    parser.add_argument("--init_checkpoint", type=str, default="None")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--caption_save_path", type=str, default="../assets/molcap/outputs.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=300)

    parser.add_argument("--text2mol_bert_path", type=str, default="../ckpts/text_ckpts/scibert_scivocab_uncased/")
    parser.add_argument("--text2mol_data_path", type=str, default="../assets/molcap/text2mol_data/")
    parser.add_argument("--text2mol_ckpt_path", type=str, default="../ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt")

    add_ddp_arguments(parser)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    if args.distributed:
        init_distributed_mode(args)
    device = torch.device(args.device)
    if args.mode == "test_from_file":
        results = test_molcap_from_file(args.caption_save_path, args, device)
        print(results)
        exit(0)

    config = json.load(open(args.config_path))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process() else logging.ERROR,
    )

    # load dataset
    if args.mode == "train":
        train_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"], split="train")
        val_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"], split="validation")
    test_dataset = SUPPORTED_MOLCAP_DATASET[args.dataset](args.dataset_path, config["data"], split="test")
    collator = MTCollator(config["data"])
    if args.distributed:
        if args.mode == "train":
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=False)
    else:
        if args.mode == "train":
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    if args.mode == "train":
        train_dataloader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, collate_fn=collator, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler, collate_fn=collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, collate_fn=collator, num_workers=args.num_workers)

    # load model
    model = SUPPORTED_MOLCAP_MODEL[config["model"]](config["network"])
    if args.init_checkpoint != "None":
        logger.info("load checkpoint from %s" % args.init_checkpoint)
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if args.param_key != "None":
            ckpt = ckpt[args.param_key]
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.mode == "train":
        train_molcap(train_dataloader, val_dataloader, test_dataloader, test_dataset, model, model_without_ddp, model_without_ddp.decoder_tokenizer, args, device)
    elif args.mode == "test":
        results = test_molcap(test_dataset, test_dataloader, model, model_without_ddp, model_without_ddp.decoder_tokenizer, args, device)
        print(results)
    elif args.mode == "traintest":
        train_molcap(train_dataloader, val_dataloader, test_dataloader, test_dataset, model, model_without_ddp, model_without_ddp.decoder_tokenizer, args, device)
        results = test_molcap(test_dataset, test_dataloader, model, model_without_ddp, model_without_ddp.decoder_tokenizer, args, device)
        print(results)
