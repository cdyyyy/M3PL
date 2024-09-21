#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=CRVPT

DATASET=$1
SEED=$2
DEVICE=$3

CFG=vit_b16_c4_ep200_batch512_5_cross_datasets
SHOTS=16

CONTRASTIVE_TYPE=supervised
LAMBDA=0.1

DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/${CONTRASTIVE_TYPE}/lambda${LAMBDA}/seed${SEED}
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
    --model-dir output/imagenet_caption/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}/${CONTRASTIVE_TYPE}/lambda${LAMBDA} \
    --load-epoch 200 \
    --eval-only
fi