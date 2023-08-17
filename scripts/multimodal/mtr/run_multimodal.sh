#!/bin/bash
MODE="zero_shot"
TASK_MODE="paragraph"
MODEL="molfm"
DEVICE=$1
EPOCHS=100

CKPT="None"
PARAM_KEY="None"
RERANK="no_rerank"

FILTER_FILE="./datasets/mtr/momu_pretrain/pair.txt"

if [ $MODEL = "molfm" ]; 
then
    CKPT="./ckpts/fusion_ckpts/molfm.pth"
    PARAM_KEY="model"
    RERANK="rerank"
elif [ $MODEL = "drugfm" ];
then
    CKPT="./ckpts/fusion_ckpts/drugfm.pth"
    PARAM_KEY="model"
    RERANK="rerank"
elif [ $MODEL = "momu" ]; 
then
    CKPT="./ckpts/fusion_ckpts/momu/MoMu-S.ckpt"
    PARAM_KEY="state_dict"
    RERANK="no_rerank"
elif [ $MODEL = "biomedgpt" ];
then
    CKPT="./ckpts/fusion_ckpts/biomedgpt-1dot6b.pth"
    PARAM_KEY="None"
    RERANK="no_rerank"
fi

mkdir ./ckpts/finetune_ckpts/mtr

python open_biomed/tasks/multi_modal_task/mtr.py \
--device ${DEVICE} \
--dataset PCdes \
--dataset_path ./datasets/mtr/PCdes \
--dataset_mode ${TASK_MODE} \
--filter \
--filter_path ${FILTER_FILE} \
--config_path ./configs/mtr/${MODEL}.json \
--init_checkpoint ${CKPT} \
--output_path ./ckpts/finetune_ckpts/mtr/${MODEL}-${TASK_MODE}-finetune.pth \
--param_key ${PARAM_KEY} \
--num_workers 1 \
--mode ${MODE} \
--patience 20 \
--epochs ${EPOCHS} \
--${RERANK} \
--rerank_num 32 \
--alpha_m2t 0.9 \
--alpha_t2m 0.9