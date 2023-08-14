#!/bin/bash
DATASET="BBBP"
DEVICE=$1

mkdir ./ckpts/finetune_ckpts/dp

python tasks/mol_task/dp.py \
--device cuda:$DEVICE \
--dataset MoleculeNet \
--dataset_path ./datasets/dp/moleculenet \
--dataset_name $DATASET \
--config_path ./configs/dp/molfm.json \
--output_path ./ckpts/finetune_ckpts/dp/molfm_finetune.pth \
--num_workers 1 \
--mode train \
--batch_size 32 \
--epochs 80 \
--patience 20 \
--seed 2