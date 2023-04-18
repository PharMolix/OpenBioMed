#!/bin/bash
ENCODER="biomedgpt"
DECODER="moflow"
DEVICE="cuda:0"

ENCODER_CKPT="None"
ENCODER_PARAM_KEY="None"

if [ $ENCODER = "biomedgpt" ]; 
then
    ENCODER_CKPT="../ckpts/fusion_ckpts/biomedgpt.ckpt"
    ENCODER_PARAM_KEY="model"
elif [ $ENCODER = "momu" ]; 
then
    ENCODER_CKPT="../ckpts/fusion_ckpts/momu/MoMu-K.ckpt"
    ENCODER_PARAM_KEY="state_dict"
fi


python tasks/multi_modal_task/text2molgen.py \
--device ${DEVICE} \
--technique z_optimize \
--encoder_config_path ./configs/text2molgen/${ENCODER}.json \
--init_encoder_checkpoint ${ENCODER_CKPT} \
--encoder_param_key ${ENCODER_PARAM_KEY} \
--decoder_config_path ./configs/text2molgen/${DECODER}.json \
--dataset prompt \
--temperature 0.05 \
--rounds_per_text 5 \
--optimize_steps 100 \
--logging_steps 100 \
--lambd 0.5 \
--evaluate \
--save_fig \
--save_path ../save/molgen \