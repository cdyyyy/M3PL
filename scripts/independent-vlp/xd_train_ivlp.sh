#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=IVLP

DATASET=imagenet
SEED=$1
DEVICE=$2

# CFG=vit_b16_c2+2_d3+3_ep2_batch4
# CFG=vit_b16_c4+3_d3+2_ep2_batch4
CFG=vit_b16_c4+4_d3+3_ep2_batch4_init
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