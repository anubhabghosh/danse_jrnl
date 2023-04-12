#!/bin/bash
for smnr in -10.0 0.0 10.0 20.0 30.0
do
    python3.7 generate_data.py \
    --n_states 2 \
    --n_obs 2 \
    --num_samples 500 \
    --sequence_length 500 \
    --sigma_e2_dB -10.0 \
    --smnr_dB $smnr \
    --dataset_type LinearSSM \
    --output_path ./data/synthetic_data/ 
done
