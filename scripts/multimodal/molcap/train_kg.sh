#!/bin/bash
MODE="train"
MODEL=$1
DEVICE=$2
EPOCHS=10

mkdir ./ckpts/finetune_ckpts/molcap/${MODEL}

CKPT="None"
PARAM_KEY="None"
if [ $MODEL = "molkformer" ];
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_stage2/checkpoint_69.pth"
    PARAM_KEY="model"
fi

python open_biomed/tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--config_path ./configs/molcap/${MODEL}.json \
--dataset chebi-kg \
--dataset_path ./datasets/chebi-kg/ \
--init_checkpoint ${CKPT} \
--param_key ${PARAM_KEY} \
--output_path ./ckpts/finetune_ckpts/molcap/${MODEL}/ \
--caption_save_path ./tmps/molcap/${MODEL}-captions.txt \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--batch_size 16 \
--logging_steps 300 \
--patience 200 \
--text2mol_bert_path ./ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ./assets/molcap/text2mol/ \
--text2mol_ckpt_path ./ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt