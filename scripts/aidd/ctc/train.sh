#!/bin/bash
DEVICE=$1

python tasks/cell_task/ctc.py \
--device ${DEVICE} \
--config_path ../configs/ctc/cellLM.json \
--dataset baron \
--dataset_path ../datasets/ctc/ \
--output_path ../ckpts/finetune_ckpts/celllm_finetune.pth \
--mode train \
--epochs 100 \
--batch_size 32 \
--logging_steps 10 \
--gradient_accumulation_steps 4 \
--patience 10
