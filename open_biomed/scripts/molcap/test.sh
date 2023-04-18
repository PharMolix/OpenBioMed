#!/bin/bash
MODE="test"
MODEL="momu"
DEVICE="cuda:0"
EPOCHS=300

python tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--config_path ./configs/molcap/${MODEL}-molt5-multnodes.json \
--dataset chebi-20 \
--dataset_path ../datasets/molcap/chebi-20 \
--output_path ../ckpts/finetune_ckpts/molcap-multnodes/checkpoints_${i}.pth \
--caption_save_path ../assets/molcap/${MODEL}-captions.txt \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--batch_size 16 \
--logging_steps 300 \
--patience 200 \
--text2mol_bert_path ../ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ../assets/molcap/text2mol_data/ \
--text2mol_ckpt_path ../ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt