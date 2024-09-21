#!/bin/bash

#cd ../..

# custom config
DATA="./datasets"
TRAINER=M3PL

DATASET=$1
SEED=$2
CFG=$3
LOADEP=$4
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only
fi