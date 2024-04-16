#!/bin/bash
DEVICE=$1

python open_biomed/tasks/multi_modal_task/molcap.py \
--device ${DEVICE} \
--dataset chebi-20 \
--dataset_path ./datasets/molcap/chebi-20 \
--caption_save_path ./assets/molcap/biot5-captions.txt \
--mode test_from_file \
--text2mol_bert_path ./ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ./assets/molcap/text2mol_data/ \
--text2mol_ckpt_path ./ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt