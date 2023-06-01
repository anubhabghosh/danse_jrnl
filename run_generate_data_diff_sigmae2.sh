#!/bin/bash
for sigma_e2_dB in -20.0 -10.0 -5.0 0.0 5.0
do
    python3.7 generate_data.py \
    --n_states 3 \
    --n_obs 3 \
    --num_samples 500 \
    --sequence_length 1000 \
    --sigma_e2_dB $sigma_e2_dB \
    --smnr_dB 10.0 \
    --dataset_type LorenzSSM \
    --output_path ./data/synthetic_data/ 
done
