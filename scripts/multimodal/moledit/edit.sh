#!/bin/bash molkformer--Graph  momu--Graph  molstm--SMILES/Graph  ID---./models/MoleculeSTM/downstream_molecule_edit_utils.py
MODE="test"
MODEL="molkformer"
DEVICE=$1
EPOCHS=100
TYPE="Graph"
ID=101

python open_biomed/tasks/mol_edit/moledit_step_02_Latent_Optimization.py \
--device ${DEVICE} \
--config_path ./configs/moledit/${MODEL}-${TYPE}-MegaMolBART.json \
--input_SMILES_file ./datasets/mol_edit/Editing_data/single_multi_property_SMILES.txt \
--language_edit_model_dir ./ckpts/finetune_ckpts/moledit/${MODEL}/${TYPE} \
--output_model_dir ./open_biomed/tasks/mol_edit \
--text_mode ./ckpts/text_ckpts/scibert_scivocab_uncased \
--epochs ${EPOCHS} \
--input_description_id ${ID} \
--MoleculeSTM_molecule_type  ${TYPE} \
--vocab_path ./ckpts/fusion_ckpts/pretrained_MegaMolBART/bart_vocab.txt \
--MegaMolBART_generation_model_dir ./ckpts/fusion_ckpts/pretrained_MegaMolBART/checkpoints \
--MASTER_PORT '6000'