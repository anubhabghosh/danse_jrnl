#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
# Import necessary libraries
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import argparse
from parse import parse
import numpy as np
import json
from utils.utils import NDArrayEncoder
import scipy
#import matplotlib.pyplot as plt
import torch
import pickle as pkl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.utils import load_saved_dataset, Series_Dataset, obtain_tr_val_test_idx, create_splits_file_name, \
    create_file_paths, check_if_dir_or_file_exists, load_splits_file, get_dataloaders, NDArrayEncoder
# Import the parameters
from config.parameters_opt import get_parameters, get_H_DANSE
#from utils.plot_functions import plot_measurement_data, plot_measurement_data_axes, plot_state_trajectory, plot_state_trajectory_axes

# Import estimator model and functions
from src.danse_supervised import DANSE_Supervised, train_danse_supervised, test_danse_supervised

def main():

    usage = "Train DANSE using trajectories of SSMs \n"\
        "python3.8 main_danse_supervised.py --mode [train/test] --model_type [gru/lstm/rnn] --dataset_mode [LinearSSM/LorenzSSM] \n"\
        "--datafile [fullpath to datafile] --splits [fullpath to splits file]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--rnn_model_type", help="Enter the desired model (rnn/lstm/gru)", type=str)
    parser.add_argument("--dataset_type", help="Enter the type of dataset (pfixed/vars/all)", type=str)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str, default=None)
    parser.add_argument("--datafile", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)
    
    args = parser.parse_args() 
    mode = args.mode
    model_type = args.rnn_model_type
    datafile = args.datafile
    dataset_type = args.dataset_type
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
    model_file_saved = args.model_file_saved
    splits_file = args.splits
    
    print("datafile: {}".format(datafile))
    print(datafile.split('/')[-1])
    # Dataset parameters obtained from the 'datafile' variable
    _, n_states, n_obs, _, T, N_samples, sigma_e2_dB, smnr_dB = parse("{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_smnr_{:f}dB.pkl", datafile.split('/')[-1])
    
    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    ssm_parameters_dict, est_parameters_dict = get_parameters(
                                            n_states=n_states,
                                            n_obs=n_obs,
                                            device=device
                                        )

    batch_size = est_parameters_dict["danse_supervised"]["batch_size"] # Set the batch size
    estimator_options = est_parameters_dict["danse_supervised"] # Get the options for the estimator

    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'generate_data.py / run_generate_data.sh' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present!")
        Z_XY = load_saved_dataset(filename=datafile)
    
    Z_XY_dataset = Series_Dataset(Z_XY_dict=Z_XY)
    ssm_model = Z_XY["ssm_model"]
    estimator_options['C_w'] = ssm_model.Cw # Get the covariance matrix of the measurement noise from the model information
    estimator_options['H'] = get_H_DANSE(type_=dataset_type, n_states=n_states, n_obs=n_obs) # Get the sensing matrix from the model info
    
    print(estimator_options['H'])
    if not os.path.isfile(splits_file):
        tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_XY_dataset,
                                                                    tr_to_test_split=0.9,
                                                                    tr_to_val_split=0.833)
        print(len(tr_indices), len(val_indices), len(test_indices))
        splits = {}
        splits["train"] = tr_indices
        splits["val"] = val_indices
        splits["test"] = test_indices
        splits_file_name = create_splits_file_name(dataset_filename=datafile,
                                                splits_filename=splits_file
                                                )
        
        print("Creating split file at:{}".format(splits_file_name))
        with open(splits_file_name, 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]

    train_loader, val_loader, test_loader = get_dataloaders(dataset=Z_XY_dataset, 
                                                            batch_size=batch_size, 
                                                            tr_indices=tr_indices, 
                                                            val_indices=val_indices, 
                                                            test_indices=test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    #ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    #print("Device Used:{}".format(device))
    
    logfile_path = "./log/"
    modelfile_path = "./models/"

    #NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_danse_supervised_opt_{}_m_{}_n_{}_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB".format(
                                                            dataset_type,
                                                            model_type,
                                                            n_states,
                                                            n_obs,
                                                            T,
                                                            N_samples,
                                                            sigma_e2_dB,
                                                            smnr_dB
                                                            )

    #print(params)
    tr_log_file_name = "training.log"
    te_log_file_name = "testing.log"

    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(os.path.join(logfile_path, main_exp_name),
                                                            file_name=tr_log_file_name)
    
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    
    flag_models_dir, _ = check_if_dir_or_file_exists(os.path.join(modelfile_path, main_exp_name),
                                                    file_name=None)
    
    print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    tr_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), tr_log_file_name)
    te_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), te_log_file_name)

    if flag_log_dir == False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)
    
    if flag_models_dir == False:
        print("Creating {}".format(os.path.join(modelfile_path, main_exp_name)))
        os.makedirs(os.path.join(modelfile_path, main_exp_name), exist_ok=True)
    
    modelfile_path = os.path.join(modelfile_path, main_exp_name) # Modify the modelfile path to add full model file 
    
    if mode.lower() == "train":
        
        model_danse = DANSE_Supervised(**estimator_options)
        tr_verbose = True  
        
        # Starting model training
        tr_losses, val_losses, _, _, _ = train_danse_supervised(
            model=model_danse,
            train_loader=train_loader,
            val_loader=val_loader,
            options=estimator_options,
            nepochs=model_danse.rnn.num_epochs,
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints="some",
            device=device,
            tr_verbose=tr_verbose
        )
        #if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
            
        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        with open(os.path.join(os.path.join(logfile_path, main_exp_name), 
            'danse_supervised_{}_losses_eps{}.json'.format(estimator_options['rnn_type'], 
            estimator_options['rnn_params_dict'][model_type]['num_epochs'])), 'w') as f:
            f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":

        #model_file_saved = "./model_checkpoints/{}_usenorm_{}_ckpt_epoch_{}.pt".format(model_type, usenorm_flag, epoch_test)
        te_loss = test_danse_supervised(
            test_loader=test_loader,
            options=estimator_options,
            device=device,
            model_file=model_file_saved,
            test_logfile_path=te_logfile_name_with_path
            )
    
    return None

if __name__ == "__main__":
    main()
