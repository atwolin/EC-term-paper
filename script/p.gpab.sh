#!/bin/bash

# Parameters
ALGORITHM="gpab"
POP_SIZE=250
CX_METHOD="cx_random"

# Function to run the experiment
run_experiment() {
    EMBEDDING_TYPE=$1
    DIM=$2
    PC=$3
    PM=$4
    MAX_GEN=$5
    echo "Running $EMBEDDING_TYPE with dim=$DIM pc=$PC pm=$PM max_gen=$MAX_GEN"
    python ../main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc $PC -pm $PM -c $CX_METHOD -g $MAX_GEN
#

export -f run_experiment
export ALGORITHM POP_SIZE CX_METHOD

# Generate the parameter combinations and run them in parallel
parallel run_experiment ::: "word2vec" "glove" "fasttext" ::: 10 15 20 25 50 100 ::: 1 0.1 0.9 ::: 0 0.03 0.1 0.5 ::: 100 150 200

# Commented out the sequential for-loops
# for EMBEDDING_TYPE in "word2vec" "glove" "fasttext"
# do
#     for DIM in 10 15 20 25 50 100
#     do
#         for PC in 1 0.1 0.9
#         do
#             for PM in 0 0.03 0.1 0.5
#             do
#                 for MAX_GEN in 100 150 200
#                 do
#                     echo "Running $EMBEDDING_TYPE with dim=$DIM pc=$PC and pm=$PM"
#                     python ../main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc $PC -pm $PM -c $CX_METHOD -g $MAX_GEN
#                 done
#             done
#         done
#     done
# done
