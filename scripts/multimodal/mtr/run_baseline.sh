#!/bin/bash
MODE="train"
TASK_MODE="paragraph"
DRUG_MODEL="graphmvp"
TEXT_MODEL="scibert"
DEVICE=$1
EPOCHS=100

FILTER_FILE="./datasets/mtr/momu_pretrain/pair.txt"

mkdir ./ckpts/finetune_ckpts/mtr

for SEED in {42..45..1}
do
echo "seed is "${SEED}
python open_biomed/tasks/multi_modal_task/mtr.py \
--device ${DEVICE} \
--dataset PCdes \
--dataset_path ./datasets/mtr/PCdes \
--dataset_mode ${TASK_MODE} \
--filter \
--filter_path ${FILTER_FILE} \
--config_path ./configs/mtr/${DRUG_MODEL}-${TEXT_MODEL}.json \
--output_path ./ckpts/finetune_ckpts/mtr/${DRUG_MODEL}-${TEXT_MODEL}-${TASK_MODE}-finetune.pth \
--num_workers 1 \
--mode ${MODE} \
--patience 20 \
--epochs ${EPOCHS} \
--no_rerank \
--seed ${SEED}
done