#!/bin/bash
MODE="zero_shot"
TASK_MODE="paragraph"
MODEL=$1
DEVICE=$2
EPOCHS=100

CKPT="None"
PARAM_KEY="None"
RERANK="no_rerank"
VIEW_OPER="ignore"
MAX_SEED=42

FILTER_FILE="./datasets/mtr/momu_pretrain/pair.txt"

if [ $MODEL = "molfm" ]; 
then
    CKPT="./ckpts/fusion_ckpts/molfm-hn/checkpoint_59.pth"
    PARAM_KEY="model"
    RERANK="rerank"
elif [ $MODEL = "drugfm" ];
then
    CKPT="./ckpts/fusion_ckpts/molfm-unimap/checkpoint_209.pth"
    PARAM_KEY="model"
    RERANK="rerank"
elif [ $MODEL = "clamp" ];
then
    CKPT="./ckpts/fusion_ckpts/clamp/checkpoint.pt"
    PARAM_KEY="model_state_dict"
    RERANK="no_rerank"
elif [ $MODEL = "momu" ]; 
then
    CKPT="./ckpts/fusion_ckpts/momu/MoMu-S.ckpt"
    PARAM_KEY="state_dict"
    RERANK="no_rerank"
elif [ $MODEL = "biomedgpt" ];
then
    CKPT="./ckpts/fusion_ckpts/biomedgpt/epoch199.pth"
    PARAM_KEY="None"
    RERANK="no_rerank"
elif [ $MODEL = "molkformer" ];
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_stage2/checkpoint_49.pth"
    PARAM_KEY="model"
    RERANK="rerank"
    VIEW_OPER="hybrid"
elif [ $MODEL = "mvmol" ];
then
    CKPT="./ckpts/fusion_ckpts/mvmol-stage3/checkpoint_7.pth"
    PARAM_KEY="model"
    RERANK="rerank"
    VIEW_OPER="hybrid"
fi

if [ $MODE = "train" ];
then
    MAX_SEED=45
fi

for (( SEED = 42; SEED <= MAX_SEED; SEED++ ));
do
echo "seed is "${SEED}
python open_biomed/tasks/multi_modal_task/mtr.py \
--device ${DEVICE} \
--dataset PCdes \
--dataset_path ./datasets/mtr/PCdes \
--dataset_mode ${TASK_MODE} \
--view_operation ${VIEW_OPER} \
--filter \
--filter_path ${FILTER_FILE} \
--config_path ./configs/mtr/${MODEL}.json \
--init_checkpoint ${CKPT} \
--output_path ./ckpts/finetune_ckpts/${MODEL}-${TASK_MODE}-finetune.pth \
--param_key ${PARAM_KEY} \
--num_workers 1 \
--mode ${MODE} \
--patience 20 \
--epochs ${EPOCHS} \
--${RERANK} \
--rerank_num 32 \
--alpha_m2t 0.8 \
--alpha_t2m 0.8
done