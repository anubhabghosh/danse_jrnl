#!/bin/bash
for N in 1000
do
    python3.7 generate_data.py \
    --n_states 3 \
    --n_obs 3 \
    --num_samples $N \
    --sequence_length 100 \
    --sigma_e2_dB -10.0 \
    --smnr_dB 10.0 \
    --dataset_type LorenzSSM \
    --output_path ./data/synthetic_data/ 
done
