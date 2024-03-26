## Running training and data-generation scripts for different parameters

This README provides an overview of the different shell scripts that can be used to run the data-driven state estimation methods such as DANSE, DANSE-Supervised (Ref. to as empirical estimation performance limit in the manuscript), 
KalmanNet and deep Markov model (DMM) for different parameters such as
- Signal-to-measurement noise ratio ($\text{SMNR}$) (in dB)
- Process noise ($\sigma_e^2$) (in dB)
- Number of training trajectories ($N$) (assumed that all trajectories are of the same length $T$)

Also includes scripts to run the data generation script (`./bin/generate_data.py`) either for different $\text{SMNR}$ values or for different values of $N$. 

The shell scripts are to be run from the main directory, i.e. `danse_jrnl`. Before the execution, the user should open the shell script and edit variables (details are found in every script as comments). Common syntax for running the shell scripts:
```bash
sh run/[run_script_name].sh
```
e.g. for running DANSE training for different $\text{SMNR}$ values, one would need to run 
```bash
sh run/run_main_danse.sh
```
The different scripts are
