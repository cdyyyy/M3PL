#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=CRVPT

DATASET=$1
SEED=$2
DEVICE=$3

CFG=vit_b16_c4_ep25_batch64_5_224_cross_datasets
SHOTS=16

CONTRASTIVE_TYPE=supervised
LAMBDA=0.2


DIR=output/${DATASET}_224_caption/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
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
    --contrastive-type ${CONTRASTIVE_TYPE} \
    --w_coeff ${LAMBDA} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi