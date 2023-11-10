#!/bin/bash molkformer--Graph  momu--Graph  molstm--SMILES/Graph
MODE="train"
MODEL="molkformer"
DEVICE=$1
EPOCHS=100
TYPE="Graph"

mkdir ./ckpts/finetune_ckpts/moledit/${MODEL}

python   open_biomed/tasks/mol_edit/moledit_step_01_Space_Alignment.py \
--device ${DEVICE} \
--MoleculeSTM_molecule_type ${TYPE} \
--config_path ./configs/moledit/${MODEL}-${TYPE}-MegaMolBART.json \
--dataset ZINC250K \
--dataset_path ./datasets/mol_edit/ZINC250K_data \
--output_path ./ckpts/finetune_ckpts/moledit/${MODEL}/${TYPE} \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 8 \
--batch_size 256 \
--vocab_path ./ckpts/fusion_ckpts/pretrained_MegaMolBART/bart_vocab.txt \
--MegaMolBART_generation_model_dir ./ckpts/fusion_ckpts/pretrained_MegaMolBART/checkpoints \
--MASTER_PORT '6000' \
--use_processed_dataset True \
--use_molecule_repr_MoleculeSTM_list_molkformer True