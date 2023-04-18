#!/bin/bash
TMP="../tmps"
SDF_FILE=${TMP}"/tmp.sdf"
OUT_FILE=${TMP}"/tmp.csv"
M2V_FILE="../assets/m2v_model.pkl"

python utils/mol_utils.py \
--mode write_sdf \
--cid2smiles_file ../assets/molcap/text2mol_data/cid_to_smiles.pkl \
--output_file $1 \
--sdf_file ${SDF_FILE}

mol2vec featurize -i ${SDF_FILE} -o ${OUT_FILE} -m ${M2V_FILE} -r 1 --uncommon UNK

python tasks/multi_modal_task/text2smigen.py \
--device $2 \
--mode test_text2mol \
--config_path ./configs/text2smi/molt5.json \
--smi_save_path $1 \
--text2mol_bert_path ../ckpts/text_ckpts/scibert_scivocab_uncased/ \
--text2mol_data_path ${TMP} \
--text2mol_ckpt_path ../ckpts/fusion_ckpts/text2mol/test_outputfinal_weights.320.pt