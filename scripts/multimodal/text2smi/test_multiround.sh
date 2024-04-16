#!/bin/bash
MODE="test"
MODEL=$1
DEVICE=$2
EPOCHS=100
SAVE_FILE=./assets/text2smi_multiround/${MODEL}-smi.txt

CKPT="None"
PARAM_KEY="None"
if [ $MODEL = "molkformer" ];
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_biot5_kg/checkpoint_18.pth"
    PARAM_KEY="model"
fi

python open_biomed/tasks/multi_modal_task/text2smigen.py \
--device ${DEVICE} \
--config_path ./configs/text2smi/${MODEL}.json \
--dataset chebi-dia \
--dataset_path ./datasets/moltextgen/chebi-dia \
--multiround \
--init_checkpoint ${CKPT} \
--param_key ${PARAM_KEY} \
--output_path ./ckpts/finetune_ckpts/text2smi_multiround/model.ckpt \
--smi_save_path ${SAVE_FILE} \
--mode ${MODE} \
--epochs ${EPOCHS} \
--eval_epochs 1 \
--warmup_epochs 5 \
--weight_decay 0 \
--num_workers 1 \
--batch_size 32 \
--logging_steps 300 \
--lr 5e-4 \
--patience 20

EVAL_TEXT2MOL=false

if $EVAL_TEXT2MOL
then
    bash ./scripts/multimodal/text2smi/evaluate_text2mol.sh ${SAVE_FILE} ${DEVICE}
fi