#!/bin/bash

# Array of dimensions
dim=(10 15 20 25 50 100)

# Loop through each dimension
for dim in "${dim[@]}"
do
    echo "Running GloVe with dimension $dim"
    ../model/glove/demo.sh python $dim
    python embedding.py all $dim
done