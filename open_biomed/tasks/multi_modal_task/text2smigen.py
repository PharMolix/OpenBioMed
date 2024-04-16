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
from torch.utils.data import DataLoader

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
RDLogger.DisableLog('rdApp.*')
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev

from open_biomed.datasets.moltextgen_dataset import SUPPORTED_TEXT2MOLGEN_DATASET
from open_biomed.models.multimodal.text2mol import Text2MolMLP
from open_biomed.models.multimodal import MolKFormer, DrugFM
from open_biomed.models.task_model.text2smi_model import Text2SMILESModel

from utils import AverageMeter, ToDevice, MTCollator

SUPPORTED_TEXT2SMI_MODEL = {
    "molt5": Text2SMILESModel,
    "chatmol": Text2SMILESModel,
    "biot5": Text2SMILESModel,
    "drugfm": DrugFM,
    "molkformer": MolKFormer,
}

def train_text2smi(train_dataset, val_dataset, test_dataset, model, decode_tokenizer, config, args, device):
    collator = MTCollator(config["data"])
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)

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

        for mol, text in train_loader:
            mol = ToDevice(mol, device)
            text = ToDevice(text, device)
            loss = model.mol_generation_loss(text, mol)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            schedular.step()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf lr=%.6lf" % (step, running_loss.get_average(), optimizer.param_groups[0]["lr"]))
                running_loss.reset()
        if not args.multiround:
            val_text2smi(val_dataset, model, config, device)
        if (epoch + 1) % args.eval_epochs == 0 and epoch / args.epochs > 0.4:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_path, "checkpoint_" + str(epoch) + ".pth"))
            if not args.multiround:
                results = test_text2smi(test_dataset, model, model.decoder_tokenizer, config, args, device)
            else:
                results = test_text2smi_dia(test_dataset, model, model.decoder_tokenizer, config, args, device)
            print(results)
    return model

def val_text2smi(val_dataset, model, config, device):
    collator = MTCollator(config["data"])
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    
    model.eval()
    val_loss = 0

    logger.info("Validating...")
    with torch.no_grad():
        for mol, text in val_loader:
            mol = ToDevice(mol, device)
            text = ToDevice(text, device)
            loss = model.mol_generation_loss(text, mol)
            val_loss += loss.detach().cpu().item()
    logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def test_text2smi(test_dataset, model, decode_tokenizer, config, args, device):
    collator = MTCollator(config["data"])
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    
    model.eval()
    outputs = []
    gts = test_dataset.smiles
    if args.output_selfies:
        gts = [Chem.MolToSmiles(Chem.MolFromSmiles(gt)) for gt in gts]

    logger.info("Testing...")
    for i, (mol, text) in enumerate(tqdm(test_loader)):
        text = ToDevice(text, device)
        output = model.decode_mol(text, num_beams=5, max_length=512)
        output = decode_tokenizer.batch_decode(output, skip_special_tokens=True)

        outputs += output
        if i <= 3:
            for j in range(1, 6):
                if args.output_selfies:
                    try:
                        import selfies as sf
                        new_output = Chem.MolToSmiles(Chem.MolFromSmiles(sf.decoder("".join(outputs[-j].split(" ")))))
                        logger.info("Generated:    %s" % new_output)
                    except:
                        logger.info("Generated:    Failed to parse output")
                else:
                    logger.info("Generated:    %s" % outputs[-j])
                logger.info("Ground truth: %s" % gts[len(outputs) - j])
                logger.info("------------------------------------------------------")

    with open(args.smi_save_path, "w") as f:
        f.write("text\tground truth\toutput\n")
        for i in range(len(outputs)):
            f.write("%s\t%s\t%s\n" % (test_dataset.texts_raw[i], gts[i], outputs[i]))

    return evaluate_generation(outputs, gts, args)

def test_text2smi_dia(test_dataset, model, decode_tokenizer, config, args, device):
    collator = MTCollator(config["data"])
    outputs = []
    gts = []

    pre_outputs = []
    for round in range(test_dataset.max_rounds):
        logger.info("Testing Round %d ..." % (round))
        cur_dataset = test_dataset.prepare_round(round, pre_outputs)
        pre_outputs = []
        loader = DataLoader(cur_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

        gts += cur_dataset.smiles
        for i, (mol, text) in enumerate(tqdm(loader)):
            text = ToDevice(text, device)
            output = model.decode_mol(text, num_beams=5, max_length=512)
            output = decode_tokenizer.batch_decode(output, skip_special_tokens=True)
            pre_outputs += output
            
            if i <= 3:
                for j in range(1, min(6, len(pre_outputs))):
                    if args.output_selfies:
                        try:
                            import selfies as sf
                            new_output = Chem.MolToSmiles(Chem.MolFromSmiles(sf.decoder("".join(output[-j].split(" ")))))
                            logger.info("Generated:    %s" % new_output)
                        except:
                            logger.info("Generated:    Failed to parse output")
                    else:
                        logger.info("Generated:    %s" % pre_outputs[-j])
                    logger.info("Ground truth: %s" % gts[len(outputs) + len(pre_outputs) - j])
                    logger.info("------------------------------------------------------")
        outputs += pre_outputs
        print("Round " + str(round) + " results: " + str(evaluate_generation(pre_outputs, gts, args)))

    return evaluate_generation(outputs, gts, args)

def evaluate_generation(outputs, gts, args):
    N = len(outputs)
    output_tokens = []
    gt_tokens = []
    levs = []
    maccs_sim, rdk_sim, morgan_sim = [], [], []
    n_bad_mols = 0
    n_exact = 0
    for i in range(N):
        try:
            if args.output_selfies:
                try:
                    import selfies as sf
                    outputs[i] = Chem.MolToSmiles(Chem.MolFromSmiles(sf.decoder("".join(outputs[i].split(" ")))))
                except:
                    output_tokens.append([])
                    gt_tokens.append([c for c in gts[i]])
                    raise ValueError
            output_tokens.append([c for c in outputs[i]])
            gt_tokens.append([[c for c in gts[i]]])
            mol_output = Chem.MolFromSmiles(outputs[i])
            mol_gt = Chem.MolFromSmiles(gts[i])
            if Chem.MolToInchi(mol_output) == Chem.MolToInchi(mol_gt):
                n_exact += 1
            maccs_sim.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol_output), MACCSkeys.GenMACCSKeys(mol_gt), metric=DataStructs.TanimotoSimilarity))
            rdk_sim.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_output), Chem.RDKFingerprint(mol_gt), metric=DataStructs.TanimotoSimilarity))
            morgan_sim.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_output, 2), AllChem.GetMorganFingerprint(mol_gt, 2)))
        except:
            n_bad_mols += 1
        levs.append(lev(outputs[i], gts[i]))

    bleu = corpus_bleu(gt_tokens, output_tokens)
    return {
        "BLEU": bleu,
        "Levenshtein": np.mean(levs),
        "Valid": 1 - n_bad_mols * 1.0 / N,
        "Exact": n_exact * 1.0 / N,
        "MACCS FTS": np.mean(maccs_sim),
        "RDKit FTS": np.mean(rdk_sim),
        "Morgan FTS": np.mean(morgan_sim),
    }

def evaluate_text2mol(args):
    text2mol = Text2MolMLP(
        ninp=768, 
        nhid=600, 
        nout=300, 
        model_name_or_path=args.text2mol_bert_path, 
        cid2smiles_path=None,
        cid2vec_path=None,
        mol2vec_output_path=os.path.join(args.text2mol_data_path, "tmp.csv")
    )
    text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path))
    device = torch.device(args.device)
    text2mol.to(device)

    logger.info("Calculating Text2Mol Metric...")
    text2mol_scores = []
    bad_smiles = 0
    with open(args.smi_save_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            line = line.rstrip("\n").split("\t")
            try:
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(line[2]))
                if smi in text2mol.smiles2vec:
                    text2mol_scores.append(text2mol(smi, line[0], device).detach().cpu().item())
            except:
                bad_smiles += 1
    logger.info("Bad SMILES: %d" % (bad_smiles))
    return np.mean(text2mol_scores)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='chebi-20')
    parser.add_argument("--dataset_path", type=str, default='../datasets/moltextgen/chebi-20')
    parser.add_argument("--multiround", action="store_true")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/text2smi/")
    parser.add_argument("--init_checkpoint", type=str, default="None")
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--smi_save_path", type=str, default="../assets/outputs.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=300)
    parser.add_argument("--text2mol_bert_path", type=str, default="")
    parser.add_argument("--text2mol_data_path", type=str, default="")
    parser.add_argument("--text2mol_ckpt_path", type=str, default="")

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
    args.output_selfies = "use_selfies" in config["network"]

    device = torch.device(args.device)

    if args.mode == "test_text2mol":
        print("Text2Mol:", evaluate_text2mol(args))
        exit(0)

    # load dataset
    if args.mode == "train":
        train_dataset = SUPPORTED_TEXT2MOLGEN_DATASET[args.dataset](args.dataset_path, config["data"], split="train")
    dataset_name = args.dataset
    if args.multiround:
        dataset_name += "-test"
    val_dataset = SUPPORTED_TEXT2MOLGEN_DATASET[dataset_name](args.dataset_path, config["data"], split="validation")
    test_dataset = SUPPORTED_TEXT2MOLGEN_DATASET[dataset_name](args.dataset_path, config["data"], split="test")

    # load model
    model = SUPPORTED_TEXT2SMI_MODEL[config["model"]](config["network"])
    if args.init_checkpoint != "None":
        logger.info("load checkpoint from %s" % args.init_checkpoint)
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if args.param_key != "None":
            ckpt = ckpt[args.param_key]
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)

    if args.mode == "train":
        train_text2smi(train_dataset, val_dataset, test_dataset, model, model.decoder_tokenizer, config, args, device)
    elif args.mode == "test":
        if os.path.exists(args.output_path):
            state_dict = torch.load(args.output_path, map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
        if not args.multiround:
            results = test_text2smi(test_dataset, model, model.decoder_tokenizer, config, args, device)
        else:
            results = test_text2smi_dia(test_dataset, model, model.decoder_tokenizer, config, args, device)
        print(results)
    elif args.mode == "traintest":
        train_text2smi(train_dataset, val_dataset, test_dataset, test_dataset, model, model.decoder_tokenizer, args, device)
        if not args.multiround:
            results = test_text2smi(test_dataset, model, model.decoder_tokenizer, config, args, device)
        else:
            results = test_text2smi_dia(test_dataset, model, model.decoder_tokenizer, config, args, device)
        print(results)