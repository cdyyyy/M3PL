#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=IVLP

DATASET=imagenet
SEED=$1
DEVICE=$2

# CFG=vit_b16_c2_ep5_batch4_4ctx_language_only
# CFG=vit_b16_c4_d12_ep5_batch4_language_only
# CFG=vit_b16_c4_d3_ep5_batch4_language_only
# CFG=vit_b16_c2_d3_ep5_batch4_language_only
# CFG=vit_b16_c4_d1_ep5_batch4_language_only
# CFG=vit_b16_c3_d2_ep5_batch4_language_only
# CFG=vit_b16_c3_d3_ep5_batch4_language_only
CFG=vit_b16_c2_d2_ep5_batch4_language_only
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi