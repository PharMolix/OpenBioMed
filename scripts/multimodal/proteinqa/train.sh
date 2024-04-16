#!/bin/bash
MODE="train"
DATASET=$1
MODEL=$2
DEVICE=$3
EPOCHS=20

mkdir ./ckpts/finetune_ckpts/proteinqa/${MODEL}

python open_biomed/tasks/multi_modal_task/proteinqa.py \
--device ${DEVICE} \
--config_path ./configs/proteinqa/${MODEL}.json \
--dataset ${DATASET} \
--dataset_path ./datasets/proteinqa/${DATASET} \
--output_path ./ckpts/finetune_ckpts/proteinqa/${MODEL}/ \
--mode ${MODE} \
--epochs ${EPOCHS} \
--gradient_accumulation_steps 1 \
--num_workers 1 \
--lr 5e-4 \
--epochs ${EPOCHS} \
--eval_epochs 2 \
--batch_size 16 \
--logging_steps 1000