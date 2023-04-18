#!/bin/bash
MODE="train"
MODEL="deepeik"
BASE="mgraphdta"
DEVICE=0
EPOCHS=500
SPLIT="warm"
Y08=true
BMKGDTI=true

if $Y08
then
    for split in ${SPLIT} 
    do
        echo "Train on Yamanishi08, split is "${split}
        CUDA_VISIBLE_DEVICES=${DEVICE} python tasks/mol_task/dti.py \
        --device cuda:0 \
        --config_path ./configs/dti/${MODEL}-${BASE}.json \
        --dataset yamanishi08 \
        --dataset_path ../datasets/dti/Yamanishi08 \
        --output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
        --mode kfold \
        --split_strategy ${split} \
        --epochs ${EPOCHS} \
        --num_workers 1 \
        --batch_size 128 \
        --lr 1e-3 \
        --logging_steps 50 \
        --patience 200
    done
fi

if $BMKGDTI
then
    for split in ${SPLIT} 
    do
        echo "Train on BMKG-DTI, split is "${split}

        CUDA_VISIBLE_DEVICES=${DEVICE} python tasks/mol_task/dti.py \
        --device cuda:0 \
        --config_path ./configs/dti/${MODEL}-${BASE}.json \
        --dataset bmkg-dti \
        --dataset_path ../datasets/dti/BMKG_DTI \
        --output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
        --mode kfold \
        --split_strategy ${split} \
        --epochs ${EPOCHS} \
        --num_workers 1 \
        --batch_size 128 \
        --lr 1e-3 \
        --logging_steps 50 \
        --patience 200
    done
fi