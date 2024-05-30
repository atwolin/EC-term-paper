#!/bin/bash

# Array of dimensions
dim=(10 15 20 25 50 100)

# Loop through each dimension
for dim in "${dim[@]}"
do
    # echo "Running GloVe with dimension $dim"
    # /home/nlplab/atwolin/EC/EC-term-paper/model/glove/demo.sh python $dim
    echo "Running Word2Vec and fastText with dimension $dim"
    python ../embedding.py $dim
done