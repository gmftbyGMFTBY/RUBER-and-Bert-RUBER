#!/bin/bash

# ========== train with weight ==========

cuda=$1
dataset=$2
model=$3

rm ./data/$dataset/result.txt

for i in {1..1}
do
    echo "========== Iteration $i begins =========="
    CUDA_VISIBLE_DEVICES=$cuda python train_unreference.py \
        --dataset $dataset
        
    CUDA_VISIBLE_DEVICES=$cuda python hybird.py \
        --mode generate \
        --dataset $dataset \
        --model $model
done