#!/bin/bash
N=5000
T=500
n_obs=2

for smnr in -10.0 0.0 10.0 20.0 30.0
do
    python3.7 generate_data.py \
    --n_states 2 \
    --n_obs ${n_obs} \
    --num_samples $N \
    --sequence_length $T \
    --sigma_e2_dB -10.0 \
    --smnr_dB $smnr \
    --dataset_type LinearSSM \
    --output_path ./data/synthetic_data/ 
done
