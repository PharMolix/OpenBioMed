#!/bin/bash
MODE="train"
DATASET=$1
MODEL=$2
DEVICE=$3
EPOCHS=100

mkdir ./ckpts/finetune_ckpts/molqa/${MODEL}

python open_biomed/tasks/multi_modal_task/molqa.py \
--device ${DEVICE} \
--config_path ./configs/molqa/${MODEL}.json \
--dataset ${DATASET} \
--dataset_path ./datasets/molqa/${DATASET} \
--output_path ./ckpts/finetune_ckpts/molqa/${MODEL}/ \
--mode ${MODE} \
--epochs ${EPOCHS} \
--gradient_accumulation_steps 16 \
--num_workers 1 \
--lr 5e-4 \
--epochs 20 \
--eval_epochs 2 \
--batch_size 8 \
--logging_steps 300