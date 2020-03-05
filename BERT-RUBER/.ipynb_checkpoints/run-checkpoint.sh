#!/bin/bash

# ========== train with weight ==========

cuda=$1
dataset=$2
model=$3
mode=$4

# ========== dailydialog ==========
# run the whole dailydialog and test the performance
# GatedGCN: 0.4092, 0.3205    **** this is the keypoint ****
# GatedGCN-role: 0.6452, 0.4827    **** this is the keypoint ****
# GatedGCN-sequential: 0.4125, 0.3242    **** this is the keypoint ****
# GatedGCN-correlation: 0.404, 0.3237    **** this is the keypoint ****
# MTGCN: 0.6818, 0.5137
# MReCoSa: 0.6212, 0.4607
# WSeq: 0.5934, 0.4398
# DSHRED: 0.5763, 0.4279
# **** what the fuck **** #
# HRED: 0.6705, 0.494

# ========== Ubuntu ==========
# GatedGCN: 0.9261, 0.6254
# MTGCN: 0.9091, 0.627
# DSHRED: 0.9187, 0.6318
# WSeq: 0.9003, 0.6229
# HRED: 0.9244， 0.6267
# MReCoSa: 0.9215, 0.6334

if [ $mode = 'train' ]; then
    rm ./data/$dataset/result.txt
    for i in {1..1}
    do
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$cuda python train_unreference.py \
            --dataset $dataset
    done
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES=$cuda python hybird.py \
            --mode generate \
            --dataset $dataset \
            --model $model
fi