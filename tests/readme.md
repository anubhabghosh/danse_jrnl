## Running evaluation scripts

This file describes the different kind of tests that could be run after training of DANSE and data-driven, comparative methods such as KalmanNet and DMM are done. The evaluation scripts are all present in `tests` directory, and should be run from the main directory of `danse_jrnl` using the general syntax:

```
[python interpreter e.g. python3.8] tests/[testing script .py file]
```

The testing scripts generally generate NMSE plots, logs and test metrics in `.json` files.

For replicating the results on 

- Linear SSM  ($2 \times 2$ linear state space model)
    - `test_kf_linear_2x2.py`: This script runs DANSE trained on the task of state estimation for the $2 \times 2$ linear, Gaussian state space model described in the paper, and compares it with a Kalman filter having full knowledge of the SSM matrices and noise statistics. 

- Lorenz SSM ($3 \times 3$ non-linear state space model)
    - `test_ukf_ekf_danse.py`: This script runs DANSE and the unsupervised KalmanNet trained on the task of state estimation for a $3 \times 3$ non-linear, Gaussian state space model known as the Lorenz attractor. The measurement matrix $\mathbf{H}_{t}$ is an identity matrix $\forall t$. 
    The comparison is made with model-based filters such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF). 
    - `test_ukf_ekf_danse_with_supervised.py`: This script runs DANSE, the empirical performance limit (referred to in the code as the DANSE_Supervised method) and the unsupervised KalmanNet trained on the task of state estimation for a $3 \times 3$ non-linear, Gaussian state space model known as the Lorenz attractor. The measurement matrix $\mathbf{H}_{t}$ is an identity matrix $\forall t$. The comparison is made with model-based filters such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF). 
    - `test_ukf_ekf_danse_sigmae.py`: This script runs DANSE and KalmanNet for the case of mismatched process noise during testing, i.e. the data-driven methods have been trained on one particular process noise ($\sigma_e^2$ in dB scale) value and SMNR, and then tested on the same SMNR but different $\sigma_e^2$ values.
    - `test_ukf_ekf_danse_sigmaw.py`: This script runs DANSE and KalmanNet for the case of mismatched measurement noise during testing, i.e. the data-driven methods have been trained on one particular process noise ($\sigma_e^2$ in dB scale) value and SMNR, and then tested on the same $\sigma_e^2$ but different SMNR values.
    - `test_ukf_ekf_danse_one_step.py`: This script runs DANSE and compares with the UKF on the task of one-step ahead of forecasting. 
    - `test_danse_diff_N.py`: This script runs DANSE with a fixed value of $T=100$ (sequence length / trajectory length) for different number of samples $N$ (number of sample trajectories) 
    - `test_ukf_ekf_danse_rnx.py`: This script runs DANSE and the unsupervised KalmanNet trained on the task of state estimation for a $n \times 3$ (where $n < 3$) non-linear, Gaussian state space model known as the Lorenz attractor. The comparison is made with model-based filters such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF). The suffix `rnx` in the name of the script indicates that the measurement matrix $\mathbf{H}_{t} = \mathbf{H} \in \mathbb{R}^{n \times 3}$ is a full-row rank matrix ($\because n < 3$) with i.i.d. Gaussian entries sampled from $\mathcal{N}(0,1)$.

- Chen SSM (Another $3 \times 3$ non-linear state space model)
    - `test_ukf_ekf_danse_Chen.py`: This script runs DANSE trained on the task of state estimation for a $3 \times 3$ non-linear, Gaussian state space model known as the Chen attractor. The comparison is made with model-based filters such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF). 

- Lorenz-96 SSM (a high-dimensional Lorenz attractor, in this case $20$-dimensional Lorenz attractor)
    - `test_ukf_ekf_danse_L96.py`: This script runs the DANSE trained on the task of state estimation for the high-dimensional Lorenz-96 attractor. In this work, we focused on a 20-dimensional Lorenz-96 attractor. The measurement matrix $\mathbf{H}_{t}$ is an 20-dim. identity matrix $\forall t$. 
    - [Optional] `test_danse_L96_diff_rn.py` and `test_danse_L96_diff_n.py`: Similar to those for the $3 \times 3$ Lorenz attractor, these scripts runs the DANSE trained on the task of state estimation for the high-dimensional Lorenz-96 attractor, where the measurement matrix $\mathbf{H}\_{t} = \mathbf{H} \in \mathbb{R}^{n \times 20}$ is a full-row rank matrix ($\because n < 20$) with i.i.d. Gaussian entries sampled from $\mathcal{N}(0,1)$ (for `*_rnx.py`) and $\mathbf{H}_{t} = \mathbf{H} \in \mathbb{R}^{n \times 20}$ is a block-identity matrix with $n < 20$ (for `*_nx.py`).

### Future work

There are some redundancies and hard-coding in the scripts in some cases. We are working to improve the code and make it more generally aplicable with less redundancies, so stay tuned for more updates! 
