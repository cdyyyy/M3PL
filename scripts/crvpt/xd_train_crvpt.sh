#!/bin/bash

#cd ../..

# custom config
DATA="../../datasets"
TRAINER=CRVPT

DATASET=imagenet
SEED=$1
DEVICE=$2

# CFG=vit_b16_c4_ep200_batch512_5_cross_datasets
# CFG=vit_b16_c4_ep200_batch720_5_cross_datasets
CFG=vit_b16_c4_ep400_batch720_5_cross_datasets_2
SHOTS=16

CONTRASTIVE_TYPE=supervised
LAMBDA=0.03
TEMPERATURE=0.1


DIR=output/${DATASET}_caption/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}/${CONTRASTIVE_TYPE}/lambda${LAMBDA}
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
    --temp ${TEMPERATURE} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi