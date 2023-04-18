#!/bin/bash
MODE="train"
MODEL="momu"
DEVICE="cuda:0"
EPOCHS=200
SAVE_FILE=../assets/text2smi/${MODEL}-smi.txt

python tasks/multi_modal_task/text2smigen.py \
--device ${DEVICE} \
--config_path ./configs/text2smi/${MODEL}-molt5.json \
--dataset chebi-20 \
--dataset_path ../datasets/molcap/chebi-20 \
--output_path ../ckpts/finetune_ckpts/text2smi/ \
--smi_save_path ${SAVE_FILE} \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--batch_size 14 \
--logging_steps 300 \
--patience 20

EVAL_TEXT2MOL=true

if $EVAL_TEXT2MOL
then
    bash scripts/text2smi/evaluate_text2mol.sh ${SAVE_FILE} ${DEVICE}
fi