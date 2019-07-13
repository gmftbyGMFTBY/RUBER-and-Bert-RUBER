#!/bin/bash

# ========== train with weight ==========
rm ./data/result.txt

for i in {1..10}
do
    echo "========== Iteration $i begins =========="
    CUDA_VISIBLE_DEVICES=0 python train_unreference.py
    CUDA_VISIBLE_DEVICES=0 python hybird.py
done

python utils.py
