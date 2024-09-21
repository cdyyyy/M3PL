#!/bin/bash

DATASET=$1
DEVICE=$2

for SEED in 1 2 3
do
    bash scripts/independent-vlp/base2new_train_ivlp.sh ${DATASET} ${SEED} ${DEVICE}
    bash scripts/independent-vlp/base2new_test_ivlp.sh ${DATASET} ${SEED} ${DEVICE}
done
