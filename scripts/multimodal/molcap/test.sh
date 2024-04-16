#!/bin/bash
MODE="test"
MODEL=$1
DEVICE=$2
EPOCHS=300
CKPT="None"
PARAM_KEY="None"

if [ ${MODEL} = "molkformer" ]; 
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_stage2/checkpoint_99.pth"
    PARAM_KEY="model"
fi
if [ ${MODEL} = "mvmol" ]; 
then
    #CKPT="./ckpts/finetune_ckpts/molcap/mvmol/checkpoint_9.pth"
    CKPT="./ckpts/fusion_ckpts/mvmol-stage3/checkpoint_9.pth"
    PARAM_KEY="model"
fi
if [ ${MODEL} = "3d-molm" ];
then
    CKPT="./ckpts/fusion_ckpts/3d_molm/stage2.ckpt"
    PARAM_KEY="state_dict"
fi 

python open_biomed/tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--config_path ./configs/molcap/${MODEL}.json \
--dataset chebi-20 \
--dataset_path ./datasets/moltextgen/chebi-20 \
--init_checkpoint ${CKPT} \
--param_key ${PARAM_KEY} \
--caption_save_path ./assets/molcap/${MODEL}-captions.txt \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--batch_size 16 \
--logging_steps 300 \
--patience 200 \
--text2mol_bert_path ./ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ./assets/molcap/text2mol_data/ \
--text2mol_ckpt_path ./ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt