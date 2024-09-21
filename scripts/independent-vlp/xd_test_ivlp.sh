#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=IVLP

DATASET=$1
SEED=$2
DEVICE=$3

# CFG=vit_b16_c2+2_d3+3_ep2_batch4
# CFG=vit_b16_c4+3_d3+2_ep2_batch4
CFG=vit_b16_c4+4_d3+3_ep2_batch4_init
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
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
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 2 \
    --eval-only
fi