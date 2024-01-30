#!/bin/bash
n_obs=2
N=5000
T=500
for smnr in -10.0 0.0 10.0 20.0 30.0
do
	python3.7 main_danse_opt.py \
	--mode train \
	--rnn_model_type gru \
	--dataset_type LorenzSSM \
	--datafile ./data/synthetic_data/trajectories_m_2_n_${n_obs}_LinearSSM_data_T_${T}_N_${N}_sigmae2_-10.0dB_smnr_$(echo $smnr)dB.pkl \
	--splits ./data/synthetic_data/splits_m_2_n_${n_obs}_LinearSSM_data_T_${T}_N_${N}_sigmae2_-10.0dB_smnr_$(echo $smnr)dB.pkl
done
