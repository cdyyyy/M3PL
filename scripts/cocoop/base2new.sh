#!/bin/bash

DATASET=$1
DEVICE=$2

for SEED in 1 2 3
do
    bash scripts/cocoop/base2new_train.sh ${DATASET} ${SEED} ${DEVICE}
    bash scripts/cocoop/base2new_test.sh ${DATASET} ${SEED} ${DEVICE}
done
