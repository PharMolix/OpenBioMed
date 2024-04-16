#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 tasks/cell_task/ctc.py \
--config_path ./configs/ctc/scbert.json \
--dataset zheng68k \
--dataset_path ../datasets/ctc/zheng68k \
--output_path ../ckpts/finetune_ckpts/scbert_finetune.pth \
--mode train \
--epochs 100 \
--batch_size 3 \
--logging_steps 2000 \
--patience 10 \
--distributed
