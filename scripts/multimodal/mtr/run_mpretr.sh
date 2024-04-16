#!/bin/bash
MODE="zero_shot"
TASK_MODE="paragraph"
MODEL=$1
DEVICE=$2
EPOCHS=100
#PERSPS=("chemical_properties_and_functions" "physical_properties" "indications" "dynamics" "actions" "metabolism" "protein-binding" "absorption" "half-life" "distribution" "elimination" "clearance" "toxicity")
PERSPS=("mix")
#PERSPS=("physical_properties")
#PERSPS=("physical_properties" "pharmacokinetic_properties" "chemical_properties_and_functions")

CKPT="None"
PARAM_KEY="None"
RERANK="no_rerank"
VIEW_OPER="ignore"
MAX_SEED=42

if [ $MODEL = "molfm_plus" ];
then
    CKPT="./ckpts/fusion_ckpts/molfm-plus/checkpoint_39.pth"
    PARAM_KEY="model"
    RERANK="no_rerank"
elif [ $MODEL = "molfm" ]; 
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
elif [ $MODEL = "molca" ];
then
    CKPT="./ckpts/fusion_ckpts/molca/stage1.ckpt"
    PARAM_KEY="state_dict"
    RERANK="rerank"
elif [ $MODEL = "3d-molm1" ];
then
    CKPT="./ckpts/fusion_ckpts/3d_molm/stage1.ckpt"
    PARAM_KEY="state_dict"
    RERANK="no_rerank"
elif [ $MODEL = "biomedgpt" ];
then
    CKPT="./ckpts/fusion_ckpts/biomedgpt/epoch199.pth"
    PARAM_KEY="None"
    RERANK="no_rerank"
elif [ $MODEL = "molkformer" ];
then
    CKPT="./ckpts/fusion_ckpts/mol_kformer_stage2/checkpoint_99.pth"
    PARAM_KEY="model"
    RERANK="rerank"
    VIEW_OPER="hybrid"
elif [ $MODEL = "mvmol" ];
then
    #CKPT="./ckpts/fusion_ckpts/mvmol-stage3-nostage1/checkpoint_9.pth"
    CKPT="./ckpts/fusion_ckpts/mvmol-stage3-nokgc/checkpoint_5.pth"
    PARAM_KEY="model"
    RERANK="no_rerank"
    VIEW_OPER="hybrid"
fi

if [ $MODE = "train" ];
then
    MAX_SEED=42
fi

for PERSP in "${PERSPS[@]}";
do
echo "perspective is "${PERSP}
for (( SEED = 42; SEED <= MAX_SEED; SEED++ ));
do
echo "seed is "${SEED}
#for (( CHECKPOINT = 29; CHECKPOINT <= 199; CHECKPOINT += 10 ));
#do
#echo "checkpoint is "${CHECKPOINT}
python open_biomed/tasks/multi_modal_task/mtr.py \
--device ${DEVICE} \
--dataset MPRetr \
--dataset_path ./datasets/mtr/mvst_mix \
--dataset_mode ${TASK_MODE} \
--perspective ${PERSP} \
--view_operation ${VIEW_OPER} \
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
#done
done
done