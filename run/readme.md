## Running training and data-generation scripts for different parameters

This README provides an overview of the different shell scripts that can be used to run the data-driven state estimation methods such as DANSE, DANSE-Supervised (Ref. to as empirical estimation performance limit in the manuscript), 
KalmanNet and deep Markov model (DMM) for different parameters such as
- Signal-to-measurement noise ratio ($\text{SMNR}$) (in dB)
- Process noise ($\sigma_e^2$) (in dB)
- Number of training trajectories ($N$) (assumed that all trajectories are of the same length $T$)

Also includes scripts to run the data generation script (`./bin/generate_data.py`) either for different $\text{SMNR}$ values or for different values of $N$. 

### Usage of shell scripts
The shell scripts are to be run from the main directory, i.e. `danse_jrnl`. Before the execution, the user should open the shell script and edit variables (details are found in every script as comments). Common syntax for running the shell scripts:
```bash
sh run/[run_script_name].sh
```
e.g. for running DANSE training for different $\text{SMNR}$ values, one would need to edit the shell script as appropriate by using `vim` / `vi`:
```bash
vi run/run_main_danse.sh
```
and once the editing is completed, save changes and then run using:
```bash
sh run/run_main_danse.sh
```

### Overview of different shell scripts for running
The different scripts in `/run/` are:
- Data-generation related
  - `run_generate_data.sh` (runs data generation for different $\text{SMNR}$ values at a specified $\sigma_e^2$ and $N$)
  - `run_generate_data_diff_N.sh` (runs data generation for different $N$ at a specified $\text{SMNR}$ and $\sigma_e^2$)
  - `run_generate_data_diff_sigmae2.sh` (runs data generation for different $\sigma_e^2$ at a specified $\text{SMNR}$ and $N$)
- Training DANSE
  - `run_main_danse.sh` (runs training for DANSE for different $\text{SMNR}$ values at a specified $\sigma_e^2$ and $N$)
  - `run_main_danse_diff_N.sh` (runs training for DANSE for different $N$ at a specified $\text{SMNR}$ and $\sigma_e^2$)
  - `run_main_danse_diff_sigmae2.sh` (runs training for DANSE for different $\sigma_e^2$ at a specified $\text{SMNR}$ and $N$)
- Training DMM
  - `run_main_dmm_causal.sh` (runs training for the causal DMM for different $\text{SMNR}$ values at a specified $\sigma_e^2$ and $N$)
- Training KalmanNet
  - `run_main_knet.sh` (runs training for KalmanNet for different $\text{SMNR}$ values at a specified $\sigma_e^2$ and $N$)
  - `run_main_knet_diff_N.sh` (runs training for KalmanNet for different $N$ at a specified $\text{SMNR}$ and $\sigma_e^2$)
