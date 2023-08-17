#!/bin/bash
DEVICE=$1

export TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=1 python tasks/mol_task/drp.py \
--device ${DEVICE} \
--task regression \
--dataset GDSC2 \
--dataset_path ../datasets/drp/GDSC2 \
--config_path ../configs/drp/deepcdr.json \
--mode train \
--num_workers 1 \
--batch_size 32