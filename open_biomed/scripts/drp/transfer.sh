#!/bin/bash
#SBATCH --job-name open-biomed
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 12:00:00 
#SBATCH --output ../logs/tdgrp_transfer.log
python tasks/mol_task/drp.py \
--device cuda:0 \
--task classification \
--dataset GDSC \
--dataset_path ../datasets/drp/GDSC \
--transfer_dataset_path ../datasets/drp/TCGA \
--config_path ./configs/drp/TGDRP_transfer.json \
--mode zero_shot_transfer \
--epochs 300