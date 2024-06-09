#!/bin/bash

# Parameters
ALGORITHM="simple_gp"
DIM=10
POP_SIZE=500
CX_METHOD="cx_simple"
MAX_EVAL=100000
EMBEDDING_TYPE="fasttext"

# Baseline
# echo "Running baseline with cr_prob=1"
# python main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc 1 -pm 0 -g $MAX_GEN -c $CX_METHOD -eval $MAX_EVAL

# Running experiments
for PC in 0 0.1 0.5 0.9
do
    for PM in 0 0.01 0.03 0.05 0.1 0.5
    do
        echo "Running $EMBEDDING_TYPE with pc=$PC and pm=$PM"
        python main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc $PC -pm $PM -c $CX_METHOD -eval $MAX_EVAL
    done
done
