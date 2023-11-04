#!/bin/bash molkformer--Graph  momu--Graph  molstm--SMILES/Graph  ID---./models/MoleculeSTM/downstream_molecule_edit_utils.py
MODE="test"
MODEL="molstm"
DEVICE=$1
EPOCHS=100
MOL_TYPE="Graph"
ID=101

python open_biomed/tasks/mol_edit/moledit_step_02_Latent_Optimization.py \
--device ${DEVICE} \
--config_path ./configs/moledit/${MODEL}-MegaMolBART.json \
--MegaMolBART_generation_model_dir ./ckpts/mol_edit_ckpts/pretrained_MegaMolBART/checkpoints \
--input_SMILES_file ./datasets/mol_edit/Editing_data/single_multi_property_SMILES.txt \
--language_edit_model_dir_new ./ckpts/finetune_ckpts/moledit/${MODEL} \
--language_edit_model_dir ./ckpts/mol_edit_ckpts/demo_checkpoints_${MOL_TYPE} \
--vocab_path ./open_biomed/models/MoleculeSTM/bart_vocab.txt \
--output_model_dir ./open_biomed/tasks/mol_edit \
--text_mode ./ckpts/mol_edit_ckpts/pretrained_SciBERT \
--epochs ${EPOCHS} \
--input_description_id ${ID} \
--MoleculeSTM_molecule_type  ${MOL_TYPE} \
--MoleculeSTM_model_dir ./ckpts/mol_edit_ckpts/demo_checkpoints_${MOL_TYPE} \
--MASTER_PORT '6000'