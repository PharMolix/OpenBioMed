#!/bin/bash
MODE="train"
MODEL=$1
DEVICE=$2
EPOCHS=100

mkdir ./ckpts/finetune_ckpts/molcap/${MODEL}

CKPT="None"
PARAM_KEY="None"
if [ $MODEL = "molkformer" ];
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_biot5.pth"
    PARAM_KEY="model"
fi
if [ $MODEL = "mvmol" ];
then
    CKPT="./ckpts/fusion_ckpts/mvmol-stage3/checkpoint_7.pth"
    PARAM_KEY="model"
    #CKPT="./ckpts/finetune_ckpts/molcap/mvmol/checkpoint_49.pth"
    #PARAM_KEY="model_state_dict"
fi

python open_biomed/tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--config_path ./configs/molcap/${MODEL}.json \
--dataset chebi-20 \
--dataset_path ./datasets/moltextgen/chebi-20 \
--init_checkpoint ${CKPT} \
--param_key ${PARAM_KEY} \
--output_path ./ckpts/finetune_ckpts/molcap/${MODEL}/ \
--caption_save_path ./assets/molcap/${MODEL}-captions.txt \
--mode ${MODE} \
--epochs ${EPOCHS} \
--warmup_epochs 1 \
--eval_epochs 10 \
--weight_decay 0 \
--num_workers 1 \
--batch_size 15 \
--gradient_accumulation_steps 1 \
--lr 5e-4 \
--logging_steps 300 \
--patience 200 \
--text2mol_bert_path ./ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ./assets/molcap/text2mol_data/ \
--text2mol_ckpt_path ./ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt