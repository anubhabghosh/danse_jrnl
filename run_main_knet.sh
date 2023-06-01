#!/bin/bash
for smnr in -10.0 0.0 10.0 20.0 30.0
do
	python3.7 main_kalmannet.py \
	--mode train \
	--knet_model_type KNetUoffline \
	--dataset_type LorenzSSMrn2 \
	--datafile ./data/synthetic_data/trajectories_m_3_n_2_LorenzSSMrn2_data_T_100_N_500_sigmae2_-10.0dB_smnr_$(echo $smnr)dB.pkl \
	--splits ./data/synthetic_data/splits_m_3_n_2_LorenzSSMrn2_data_T_100_N_500_sigmae2_-10.0dB_smnr_$(echo $smnr)dB.pkl
done
