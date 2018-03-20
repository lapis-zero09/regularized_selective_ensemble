#!/bin/bash


# matlabファイルの実行
# nohup echo " weight_for_train_data()" | matlab -nodisplay > weight.out &

# rseのlambda決定
lambda=$(python lambda_decision.py | tail -n 1 >&1)

# matlabファイルの実行
eval "matlab -nodesktop -nosplash -r \"weight_for_test_data($lambda)\" > weight.out"
