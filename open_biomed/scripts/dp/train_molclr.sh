#!/bin/bash
DATASET="BBBP"

python tasks/mol_task/dp.py \
--device cuda:0 \
--dataset MoleculeNet \
--dataset_path ../datasets/dp/moleculenet \
--dataset_name ${DATASET} \
--config_path ./configs/dp/molclr.json \
--num_workers 1 \
--mode train \
--batch_size 32 \
--epochs 100 \
--patience 10