#!/bin/bash

# Parameters
ALGORITHM="rf"
POP_SIZE=500
CX_METHOD="cx_random"
MAX_EVAL=20000
# EMBEDDING_TYPE="fasttext"

# Baseline
# echo "Running baseline with cr_prob=1"
# python main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc 1 -pm 0 -g $MAX_GEN -c $CX_METHOD -eval $MAX_EVAL

# Running experiments
for EMBEDDING_TYPE in "word2vec" "glove" "fasttext"
do
    for DIM in 10 15 20 25 50 100
    do
        for PC in 1 0.1 0.9
        do
            for PM in 0.1 0 0.03  0.5
            do
                echo "Running $EMBEDDING_TYPE with dim=$DIM pc=$PC and pm=$PM"
                python ../main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc $PC -pm $PM -c $CX_METHOD -eval $MAX_EVAL
            done
        done
    done
done

# #!/bin/bash

# # Parameters
# ALGORITHM="rf"
# POP_SIZE=500
# CX_METHOD="cx_random"
# MAX_EVAL=20000

# # Function to run the experiment
# run_experiment() {
#     EMBEDDING_TYPE=$1
#     DIM=$2
#     PC=$3
#     PM=$4
#     echo "Running $EMBEDDING_TYPE with dim=$DIM pc=$PC and pm=$PM"
#     python ../main.py -algo $ALGORITHM -e $EMBEDDING_TYPE -n $DIM -p $POP_SIZE -pc $PC -pm $PM -c $CX_METHOD -eval $MAX_EVAL
# }

# export -f run_experiment
# export ALGORITHM POP_SIZE CX_METHOD MAX_EVAL

# # Generate the parameter combinations and run them in parallel
# parallel run_experiment ::: "word2vec" "glove" "fasttext" ::: 10 15 20 25 50 100 ::: 1 0.1 0.9 ::: 0 0.03 0.1 0.5
