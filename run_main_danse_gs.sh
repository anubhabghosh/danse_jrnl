#!/bin/bash
for r in 10.0
do
	python3.7 main_danse_gs.py \
	--mode train \
	--rnn_model_type gru \
	--dataset_type LorenzSSM \
	--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_sigmae2_-10.0dB_smnr_$(echo $r)dB.pkl \
	--splits ./data/synthetic_data/splits_m_3_n_3_LorenzSSM_data_T_1000_N_500_sigmae2_-10.0dB_smnr_$(echo $r)dB.pkl
done
