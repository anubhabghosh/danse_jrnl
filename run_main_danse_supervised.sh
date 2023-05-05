#!/bin/bash
python3.8 main_danse_supervised.py \
--mode train \
--rnn_model_type gru \
--dataset_type LinearSSM \
--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_sigmae2_-10.0dB_smnr_-10.0dB.pkl \
--splits ./data/synthetic_data/splits.pkl