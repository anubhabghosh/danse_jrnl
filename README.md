# DANSE: Data-driven Non-linear State Estimation of Model-free Process Using Unsupervised Learning

### Under review in IEEE-TSP (June 2023)

## Authors
Anubhab Ghosh (anubhabg@kth.se), Antoine Honoré (honore@kth.se), Saikat Chatterjee (sach@kth.se)

## Dependencies
- PyTorch (1.6.0)
- Python (>= 3.7.0) with standard packages as part of an Anaconda installation such as Numpy, Scipy, Matplotlib etc.
- Filterpy (1.4.5) (for implementation of Unscented Kalman Filter (UKF)): [https://filterpy.readthedocs.io/en/latest/](https://filterpy.readthedocs.io/en/latest/)
- Jupyter notebook (>= 6.4.6) (for result analysis)

## Reference models (implemented in PyTorch)

- Kalman filter (KF)
- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- Unsupervised KalmanNet
    - The code was adopted from the repository of the authors: [https://github.com/KalmanNet/Unsupervised_EUSIPCO_22](https://github.com/KalmanNet/Unsupervised_EUSIPCO_22)
    - Experimental details taken also from the repository of the supervised KalmanNet: [https://github.com/KalmanNet/KalmanNet_TSP](https://github.com/KalmanNet/KalmanNet_TSP)

## Code organization

This would be required organization of files and folders for reproducing results. If certain folders are not present, they should be created at that level.

````
- data/ (contains stored datasets in .pkl files)
- src/ (contains model related files)
| - danse.py 
| - ekf.py
|···
- log/ (contains training and evaluation logs, losses in `.json`, `.log` files)
- models/ (contains saved model checkpoints saved as `.pt` files)
- figs/ (contains resulting model figures)
- utils/ (contains helping functions for \src\, etc.)
- figs/ (contains result figures)
- tests/ (contains files and functions for evaluation at test time)
- parameters.py (Pythnon file containing relevant parameters for different architectures)
- main_danse.py (main function for calling training 'DANSE' model)
- main_kalmannet.py (main function for calling reference training 'KalmanNet' model)
- ssm_models.py (contains code for implementing state space models)
- generate_data.py (contains code for generating training datasets)
````

## Brief outline of DANSE training

1. Generate data by calling `generate_data.py`. This can be done in a simple manner by editing and calling the shell script `run_generate_data.sh`. Data gets stored at `data/synthetic_data/`. For e.g. to generate trajectory data with 500 samples with each trajetcoyr of length 1000, from a Lorenz Attractor model (m=3, n=3), with $\frac{1}{r^2}= 20$ dB, and $\nu=$-20 dB, the syntax should be 
````
[python interpreter e.g. python3.8] generate_data.py --n_states 3 --n_obs 3 --num_samples 500 --sequence_length 1000 --inverse_r2_dB 20 --nu_dB -20 --dataset_type LorenzSSM --output_path [dataset location e.g. ./data/synthetic_data/] \
````

2. Edit the parameters as per user choice to set architecture for DANSE in `parameters.py`.

3. Run the training for DANSE by calling `main_danse.py`. This can be done in a simple manner by editing and calling the shell script 
`run_main_danse.sh`. Ensure that directories `/log/` and `/models/` have been created. E.g. to run a DANSE model employing a GRU architecture as the RNN, using the Lorenz attractor dataset as described above, the syntax should be 
```
python3.8 main_danse.py \
--mode train \
--rnn_model_type gru \
--dataset_type LorenzSSM \
--datafile [full path to dataset, e.g. ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_r2_40.0dB_nu_-20dB.pkl] \
--splits ./data/synthetic_data/splits.pkl
```

4. Run the training for the unsupervised KalmanNet by calling `main_kalmannet.py`. Also posible in similar manner as `run_main_knet_gpu1.sh`. Parameters have to be edited in `parameters.py`.

### Grid-search (for architectural choice of DANSE)

Can be run by calling the script `main_danse_gs.py` with grid-search parameters to be edited in the script directly. Relevant shell script that can be edited and used: `run_main_danse_gs_gpu1.sh`.

## Evaluation

Once files are created, the evaluation can be done by calling scripts in `/tests/`. Paths to model files and log files should be edited in the script directly. 

1. Linear model ($2 \times 2$ model) comparison: `/tests/test_kf_linear_2x2.py`
2. Lorenz model comparsion: `/tests/test_ukf_ekf_danse.py`.

