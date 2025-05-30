#!/bin/bash

pred_lens=(12 96 192 336 720)

exp_names=TimeSeriesConfig
for len in "${pred_lens[@]}"
do
  echo "run_train.py --exp_name "$exp_names" --retrain 1 --pred_len $len"
  python -u run_train.py --exp_name "$exp_names" --retrain 1 --pred_len "$len"
done