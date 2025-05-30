#!/bin/bash

densities=(0.02 0.04 0.06 0.08 0.10)

exp_names=(NTCConfig)
for len in "${densities[@]}"
  do
    echo "./run_train.py --exp_name "$exp_names" --retrain 1 --density $len"
    python -u ../run_train.py --exp_name "$exp_names" --retrain 1 --density "$len"
  done