#!/bin/bash
PYTHON="python3.8"
n_states=3
n_obs=3
N=100
T=25
dataset_type="LorenzSSM"
sigma_e2_dB=-10.0
model_type="gru"
script_name="main_dmm_causal.py"

for smnr in -10.0 0.0 10.0 20.0 30.0
do
	datafile="./data/synthetic_data/trajectories_m_${n_states}_n_${n_obs}_${dataset_type}_data_T_${T}_N_${N}_sigmae2_${sigma_e2_dB}dB_smnr_$(echo $smnr)dB.pkl"
	splitsfile="./data/synthetic_data/splits_m_${n_states}_n_${n_obs}_${dataset_type}_data_T_${T}_N_${N}_sigmae2_${sigma_e2_dB}dB_smnr_$(echo $smnr)dB.pkl"
	${PYTHON} ${script_name} \
	--mode train \
	--rnn_model_type ${model_type} \
	--dataset_type ${dataset_type} \
	--datafile ${datafile} \
	--splits ${splitsfile}
done