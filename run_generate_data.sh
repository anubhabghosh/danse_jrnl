#!/bin/bash
for smnr in -10.0
do
    python3.8 generate_data.py \
    --n_states 3 \
    --n_obs 3 \
    --num_samples 500 \
    --sequence_length 1000 \
    --sigma_e2_dB -10.0 \
    --smnr_dB $smnr \
    --dataset_type LinearSSM \
    --output_path ./data/synthetic_data/ 
done