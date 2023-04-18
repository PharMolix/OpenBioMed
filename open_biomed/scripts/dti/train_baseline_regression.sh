#!/bin/bash
MODE="train"
MODEL="deepdta"
DEVICE="cuda:0"
EPOCHS=300

python tasks/mol_task/dti.py \
--device ${DEVICE} \
--config_path ./configs/dti/${MODEL}.json \
--dataset davis \
--dataset_path ../datasets/dti/davis \
--output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
--mode train \
--epochs ${EPOCHS} \
--num_workers 4 \
--batch_size 128 \
--lr 1e-3 \
--logging_steps 50 \
--patience 20