#!/bin/bash

# Parameters
ALGORITHM="simple_gp"
POP_SIZE=500
CX_METHOD="cx_simple"
MAX_EVAL=20000
EMBEDDING_TYPE="glove"

# Baseline
# echo "Running baseline with cr_prob=1"
# python main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc 1 -pm 0 -g $MAX_GEN -c $CX_METHOD -eval $MAX_EVAL

# Running experiments
#for DIM in 10 15 20 25 50 100
#do
for PC in 1 0.1 0.5
do
    for PM in 0 0.03 0.1 0.5
    do
        echo "Running $EMBEDDING_TYPE with dim=$DIM pc=$PC and pm=$PM"
        python ../main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc $PC -pm $PM -c $CX_METHOD -eval $MAX_EVAL
    done
done
#done
