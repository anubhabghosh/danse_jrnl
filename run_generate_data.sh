#!/bin/bash
# This script is used to run the generate_data.py file for creating training data.
# Creator: Anubhab Ghosh, Nov 2023.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python3.8"

# The number of i.i.d. trajectories each of length T that constitute the training data
N=500

# Length of each such training data trajectory, default it is set to T=1000
T=100

# Number of hidden states in a dynamical system, usually for Lorenz (a.k.a. Lorenz-63), Chen and Rossler
# attractors, the number of hidden states is equal to 3, while for Lorenz-96, this value must be changed to
# n_states= 20 (currently hardcoded in this manner) but can be in general n_states >= 4
n_states=2

# Number of observations in a dynamical system
n_obs=2

# dataset_type defines the type of dynamical system, the general terminology, e.g. for the Lorenz 63 system, 
# the type is LorenzSSM, similarly for Chen and Rossler attractor we have ChenSSM and RosslerSSM respectively.
# For the Lorenz-96 model, we have Lorenz96SSM.
# For underdetermined scenario with random matrix subsampled measurements: LorenzSSMrn${n_obs}, ChenSSMrn${n_obs},
# RosslerSSMrn${n_obs}, Lorenz96SSMrn${n_obs} with deterministic measurements: LorenzSSMn${n_obs}, ChenSSMn${n_obs}, 
# RosslerSSMn${n_obs}, Lorenz96SSMn${n_obs}.
# For the linear system, we have LinearSSM (can handle both full-rank, deterministic downsmapled case).
dataset_type="LinearSSM"

# The name of the script for generating data with full path name
script_name="generate_data.py"

# Output path to store the data
output_path="./data/synthetic_data/"

# Set the process noise level (in dB)
sigma_e2_dB=-10.0

# For different signal-to-measurement-noise ratio (SMNRs), run the data generation 
for smnr in -10.0 0.0 10.0 20.0 30.0
do
    ${PYTHON} ${script_name} \
    --n_states ${n_states} \
    --n_obs ${n_obs} \
    --num_samples $N \
    --sequence_length $T \
    --sigma_e2_dB $sigma_e2_dB \
    --smnr_dB $smnr \
    --dataset_type ${dataset_type} \
    --output_path ${output_path}
done