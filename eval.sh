#!/bin/bash

fold_num=$1
model_path="weights/model_weights_b3_fold$fold_num.pth"
timestamp=$(TZ=KST date)
echo "Date : $timestamp"
echo "Evaluate fold No. $fold_num ... "
echo "Load trained model : $model_path ..."

python train.py --fold $fold_num --trained_model=$model_path  2>&1 | tee log/eval_log_fold${fold_num}.log