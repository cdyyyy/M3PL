#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=CoCoOp
# TRAINER=CoOp

DATASET=imagenet
SEED=$1
DEVICE=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1  # uncomment this when TRAINER=CoCoOp and DATASET=imagenet
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
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