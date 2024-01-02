#!/bin/bash
MODE="train"
NUM_GPUS=4
MODEL=$1
EPOCHS=75

mkdir ./ckpts/finetune_ckpts/molcap/${MODEL}

CKPT="None"
PARAM_KEY="None"
if [ $MODEL = "molkformer" ];
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_stage2/checkpoint_49.pth"
    PARAM_KEY="model"
fi

CUDA_VISIBLE_DEVICES=0,1,3,7 \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} open_biomed/tasks/multi_modal_task/molcap.py \
--config_path ./configs/molcap/${MODEL}.json \
--dataset chebi-20 \
--dataset_path ./datasets/molcap/chebi-20 \
--init_checkpoint ${CKPT} \
--param_key ${PARAM_KEY} \
--output_path ./ckpts/finetune_ckpts/molcap/${MODEL}/ \
--caption_save_path ./assets/molcap/${MODEL}-captions.txt \
--mode ${MODE} \
--epochs ${EPOCHS} \
--eval_epochs 10 \
--warmup_epochs 5 \
--weight_decay 0 \
--num_workers 1 \
--batch_size 16 \
--lr 5e-4 \
--logging_steps 100 \
--patience 200 \
--text2mol_bert_path ./ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ./assets/molcap/text2mol_data/ \
--text2mol_ckpt_path ./ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt \
--distributed