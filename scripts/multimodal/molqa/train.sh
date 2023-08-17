#!/bin/bash
MODE="train"
MODEL="graphmvp_molt5"
DEVICE=$1
EPOCHS=100

mkdir ./ckpts/finetune_ckpts/molqa/${MODEL}

python open_biomed/tasks/multi_modal_task/molqa.py \
--device ${DEVICE} \
--config_path ./configs/molqa/${MODEL}.json \
--dataset chembl-qa \
--dataset_path ./datasets/molqa/ChEMBL-QA \
--output_path ./ckpts/finetune_ckpts/molqa/${MODEL}/ \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--lr 1e-4 \
--epochs 50 \
--batch_size 32 \
--logging_steps 300