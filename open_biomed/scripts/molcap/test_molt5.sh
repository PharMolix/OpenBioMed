#!/bin/bash
MODE="test"
DEVICE="cuda:0"

python tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--config_path ./configs/molcap/molt5.json \
--dataset chebi-20 \
--dataset_path ../datasets/molcap/chebi-20 \
--caption_save_path ../assets/molcap/molt5-captions-test.txt \
--mode ${MODE} \
--num_workers 1 \
--batch_size 16 \
--text2mol_bert_path ../ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ../assets/molcap/text2mol_data/ \
--text2mol_ckpt_path ../ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt