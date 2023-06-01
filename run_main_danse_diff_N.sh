#!/bin/bash
smnr=10.0
m=3
n=3
model="LorenzSSM"
T=100
sigmae2=-10.0
rnn_type="gru"

for N in 1000
do
	python3.7 main_danse_opt.py \
	--mode train \
	--rnn_model_type ${rnn_type} \
	--dataset_type ${model} \
	--datafile ./data/synthetic_data/trajectories_m_${m}_n_${n}_${model}_data_T_${T}_N_${N}_sigmae2_${sigmae2}dB_smnr_$(echo $smnr)dB.pkl \
	--splits ./data/synthetic_data/splits_m_${m}_n_${n}_${model}_data_T_${T}_N_${N}_sigmae2_${sigmae2}dB_smnr_$(echo $smnr)dB.pkl
done
