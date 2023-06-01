#!/bin/bash
smnr=10.0
for N in 5 10 20 40 80
do
	python3.7 main_kalmannet.py \
	--mode train \
	--knet_model_type KNetUoffline \
	--dataset_type LorenzSSM \
	--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_100_N_${N}_sigmae2_-10.0dB_smnr_$(echo $smnr)dB.pkl \
	--splits ./data/synthetic_data/splits_m_3_n_3_LorenzSSM_data_T_100_N_${N}_sigmae2_-10.0dB_smnr_$(echo $smnr)dB.pkl
done
