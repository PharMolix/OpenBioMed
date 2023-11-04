#!/bin/bash molkformer--Graph  momu--Graph  molstm--SMILES/Graph
MODE="train"
MODEL="molkformer"
DEVICE=$1
EPOCHS=100
MOL_TYPE="Graph"

mkdir ./ckpts/finetune_ckpts/moledit/${MODEL}

python   open_biomed/tasks/mol_edit/moledit_step_01_Space_Alignment.py \
--device ${DEVICE} \
--MoleculeSTM_molecule_type ${MOL_TYPE} \
--config_path ./configs/moledit/${MODEL}-MegaMolBART.json \
--dataset ZINC250K \
--dataset_path ./datasets/mol_edit/ZINC250K_data \
--output_path ./ckpts/finetune_ckpts/moledit/${MODEL}/ \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 8 \
--batch_size 256 \
--vocab_path ./open_biomed/models/MoleculeSTM/bart_vocab.txt \
--MoleculeSTM_model_dir ./ckpts/mol_edit_ckpts/demo_checkpoints_${MOL_TYPE} \
--MegaMolBART_generation_model_dir ./ckpts/mol_edit_ckpts/pretrained_MegaMolBART/checkpoints \
--MASTER_PORT '6000'