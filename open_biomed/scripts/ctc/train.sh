#!/bin/bash
DEVICE=0

python tasks/cell_task/ctc.py \
--device ${DEVICE} \
--config_path ./configs/ctc/scbert.json \
--dataset zheng68k \
--dataset_path ../datasets/ctc/zheng68k \
--output_path ../ckpts/finetune_ckpts/scbert_finetune.pth \
--mode train \
--epochs 100 \
--batch_size 3 \
--logging_steps 2000 \
--patience 10
