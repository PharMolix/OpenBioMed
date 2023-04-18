#!/bin/bash
#SBATCH --job-name open-biomed
#SBATCH --array 0
#SBATCH --gres gpu:a100:1
#SBATCH --time 12:00:00 
#SBATCH --output ../logs/tdgrp_gdsc.log
python tasks/mol_task/drp.py \
--device cuda:0 \
--task regression \
--dataset GDSC \
--dataset_path ../datasets/drp/GDSC \
--config_path ./configs/drp/TGDRP.json \
--mode train