#!/bin/bash
MODE="test"
DEVICE="cuda:0"

GEN_SMILES=true
SAVE_FILE=../assets/text2smi/molt5-smi.txt

if $GEN_SMILES
then
    python tasks/multi_modal_task/text2smigen.py \
    --device ${DEVICE} \
    --config_path ./configs/text2smi/molt5.json \
    --output_path None \
    --dataset chebi-20 \
    --dataset_path ../datasets/molcap/chebi-20 \
    --smi_save_path ${SAVE_FILE} \
    --mode ${MODE} \
    --num_workers 1 \
    --batch_size 16
fi

EVAL_TEXT2MOL=true

if $EVAL_TEXT2MOL
then
    bash scripts/text2smi/evaluate_text2mol.sh ${SAVE_FILE} ${DEVICE}
fi