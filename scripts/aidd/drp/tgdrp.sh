#!/bin/bash
DEVICE=$1

python open_biomed/tasks/mol_task/drp.py \
--device $DEVICE \
--task regression \
--dataset GDSC \
--dataset_path ./datasets/drp/GDSC \
--config_path ./configs/drp/TGDRP.json \
--mode train