#!/bin/bash
DEVICE=$1
MODEL="gnn_ppi"
DATASET="SHS27k"
SPLIT="dfs"

mkdir ./ckpts/finetune_ckpts/ppi/

python open_biomed/tasks/prot_task/ppi.py \
--device ${DEVICE} \
--mode train \
--config_path ./configs/ppi/${MODEL}.json \
--dataset ${DATASET} \
--dataset_path ./datasets/ppi/${DATASET} \
--split_strategy ${SPLIT} \
--output_path ./ckpts/finetune_ckpts/ppi/${MODEL}.pth \
--num_workers 1 \
--epochs 200 \
--patience 50 \
--lr 0.001 \
--logging_steps 15 \
--batch_size 512 \
--seed 42
