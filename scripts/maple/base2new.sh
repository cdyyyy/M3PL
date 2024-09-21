#!/bin/bash

DATASET=$1
DEVICE=$2

for SEED in 1 2 3
do
    bash scripts/maple/base2new_train_maple.sh ${DATASET} ${SEED} ${DEVICE}
    bash scripts/maple/base2new_test_maple.sh ${DATASET} ${SEED} ${DEVICE}
done
