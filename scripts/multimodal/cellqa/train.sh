#!/bin/bash
MODE="train"
MODEL="geneformer_biot5"
DEVICE=$1
EPOCHS=100

mkdir ./ckpts/finetune_ckpts/cellqa/${MODEL}

python open_biomed/tasks/multi_modal_task/cellqa.py \
--device ${DEVICE} \
--config_path ./configs/cellqa/${MODEL}.json \
--dataset cqa \
--dataset_path ./datasets/cellqa/cqa \
--output_path ./ckpts/finetune_ckpts/cellqa/${MODEL}/ \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--lr 1e-4 \
--epochs 10 \
--eval_epochs 1 \
--batch_size 32 \
--logging_steps 1000