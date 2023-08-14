#!/bin/bash
DATASET="BBBP"
DEVICE=$1

mkdir ./ckpts/finetune_ckpts/dp

python open_biomed/tasks/mol_task/dp.py \
--device $DEVICE \
--dataset MoleculeNet \
--dataset_path ./datasets/dp/moleculenet \
--dataset_name $DATASET \
--config_path ./configs/dp/graphmvp.json \
--output_path ./ckpts/finetune_ckpts/dp/graphmvp_finetune_$DATASET.pth \
--num_workers 1 \
--mode train \
--batch_size 256 \
--epochs 80 \
--patience 20

