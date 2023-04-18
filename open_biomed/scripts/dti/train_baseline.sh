#!/bin/bash
MODE="train"
MODEL="mgraphdta"
SPLIT="warm cold_drug cold_protein cold_cluster"
DEVICE="cuda:0"
EPOCHS=300
Y08=false
BMKGDTI=true

if $Y08
then
    for split in ${SPLIT} 
    do
        echo "Train on Yamanishi08, split is "${split}

        python tasks/mol_task/dti.py \
        --device ${DEVICE} \
        --config_path ./configs/dti/${MODEL}.json \
        --dataset yamanishi08 \
        --dataset_path ../datasets/dti/Yamanishi08 \
        --output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
        --mode kfold \
        --split_strategy ${split} \
        --epochs ${EPOCHS} \
        --num_workers 4 \
        --batch_size 128 \
        --lr 1e-3 \
        --logging_steps 50 \
        --patience 50
    done
fi

if $BMKGDTI
then
    for split in ${SPLIT} 
    do
        echo "Train on BMKG-DTI, split is "${split}

        python tasks/mol_task/dti.py \
        --device ${DEVICE} \
        --config_path ./configs/dti/${MODEL}.json \
        --dataset bmkg-dti \
        --dataset_path ../datasets/dti/BMKG_DTI \
        --output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
        --mode kfold \
        --split_strategy ${split} \
        --epochs ${EPOCHS} \
        --num_workers 4 \
        --batch_size 128 \
        --lr 1e-3 \
        --logging_steps 50 \
        --patience 50
    done
fi