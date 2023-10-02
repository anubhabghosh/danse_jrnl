#!/bin/bash
N=100
T=500
n_states=20
n_obs=20
sigma_e2_dB=-10.0
model_type="Lorenz96SSM"

for smnr in 10.0
do
    python3.8 generate_data.py \
    --n_states ${n_states} \
    --n_obs ${n_obs} \
    --num_samples $N \
    --sequence_length $T \
    --sigma_e2_dB ${sigma_e2_dB} \
    --smnr_dB $smnr \
    --dataset_type ${model_type} \
    --output_path ./data/synthetic_data/ 
done
