#!/bin/bash

DATASET=$1
DEVICE=$2

for SEED in 1 2 3
do
    bash scripts/language-prompting/base2new_train_lp.sh ${DATASET} ${SEED} ${DEVICE}
    bash scripts/language-prompting/base2new_test_lp.sh ${DATASET} ${SEED} ${DEVICE}
done
