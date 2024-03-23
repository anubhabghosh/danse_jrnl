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
from config.parameters_opt import get_parameters, J_test, delta_t_test, delta_t, J_gen, A_fn, h_fn, get_H_DANSE
from bin.ssm_models import *
#from utils.plot_functions import plot_measurement_data, plot_measurement_data_axes, plot_state_trajectory, plot_state_trajectory_axes

# Import estimator model and functions
from src.k_net import KalmanNetNN, train_KalmanNetNN, test_KalmanNetNN

def f_lorenz_danse(x, device='cpu'):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor).to(device)
    C = torch.Tensor([[-10, 10,    0],
                      [ 28, -1,    0],
                      [  0,  0, -8/3]]).type(torch.FloatTensor).to(device)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    #delta_t = 0.02 # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3).type(torch.FloatTensor).to(device)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

def main():

    usage = "Train DANSE using trajectories of SSMs \n"\
        "python3.8 main_kalmannet.py --mode [train/test] --knet_model_type [gru/lstm/rnn] --dataset_mode [LinearSSM/LorenzSSM] \n"\
        "--datafile [fullpath to datafile] --splits [fullpath to splits file]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--knet_model_type", help="Enter the desired model (default: KNetUoffline)", type=str)
    parser.add_argument("--dataset_type", help="Enter the type of dataset (pfixed/vars/all)", type=str)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str, default=None)
    parser.add_argument("--datafile", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)
    
    args = parser.parse_args() 
    mode = args.mode
    knet_model_type = args.knet_model_type # For unsupervised, we need this to be: "KNetUoffline"
    datafile = args.datafile
    dataset_type = args.dataset_type
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
    model_file_saved = args.model_file_saved
    splits_file = args.splits
    
    print("datafile: {}".format(datafile))
    print(datafile.split('/')[-1])
    # Dataset parameters obtained from the 'datafile' variable
    _, n_states, n_obs, ssm_type, T, N_samples, sigma_e2_dB, smnr_dB = parse("{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_smnr_{:f}dB.pkl", datafile.split('/')[-1])
    
    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))



    ssm_parameters_dict, est_parameters_dict = get_parameters(
                                            n_states=n_states,
                                            n_obs=n_obs,
                                            device=device
                                        )

    batch_size = est_parameters_dict[knet_model_type]["batch_size"] # Set the batch size
    estimator_options = est_parameters_dict[knet_model_type] # Get the options for the estimator
    H_ = get_H_DANSE(type_=ssm_type, n_states=n_states, n_obs=n_obs)
    val_batch_size = estimator_options["N_CV"]
    te_batch_size = estimator_options["N_T"]
    
    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'generate_data.py / run_generate_data.sh' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present!")
        Z_XY = load_saved_dataset(filename=datafile)
    
    Z_XY_dataset = Series_Dataset(Z_XY_dict=Z_XY)

    if not os.path.isfile(splits_file):
        tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_XY_dataset,
                                                                    tr_to_test_split=0.66667,
                                                                    tr_to_val_split=0.5)
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
                                                            test_indices=test_indices, 
                                                            val_batch_size=val_batch_size,
                                                            te_batch_size=te_batch_size)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    #ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    #print("Device Used:{}".format(device))
    
    logfile_path = "./log/"
    modelfile_path = "./models/"

    #NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_{}_m_{}_n_{}_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB".format(
                                                            dataset_type,
                                                            knet_model_type,
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
    
    if ssm_type == "LinearSSM":
        ssm_model = LinearSSM(n_states=n_states, n_obs=n_obs,
                        mu_e=np.zeros((n_states,)), mu_w=np.zeros((n_obs,)))
        def fn(x):
            return torch.from_numpy(ssm_model.F).type(torch.FloatTensor).to(device) @ x
        
        def hn(x):
            return torch.from_numpy(ssm_model.H).type(torch.FloatTensor).to(device) @ x

    elif ssm_type == "LorenzSSM":
        ssm_model = LorenzSSM(n_states=n_states,
                            n_obs=n_states,
                            J=J_gen, 
                            delta=delta_t,
                            delta_d=delta_t,
                            alpha=ssm_parameters_dict[ssm_type]["alpha"], 
                            decimate=False,
                            mu_e=np.zeros((n_states,)),
                            mu_w=np.zeros((n_obs,)),
                            use_Taylor=ssm_parameters_dict[ssm_type]["use_Taylor"])

        def fn(x):
            return f_lorenz_danse(x, device=device)
        
        def hn(x):
            return h_fn(x)
    
    elif ssm_type == "LorenzSSMn2" or "LorenzSSMn1":

        ssm_model = LorenzSSM(n_states=n_states,
                            n_obs=n_obs,
                            J=J_gen, 
                            delta=delta_t,
                            delta_d=delta_t,
                            alpha=ssm_parameters_dict[ssm_type]["alpha"], 
                            decimate=False,
                            H=H_,
                            mu_e=np.zeros((n_states,)),
                            mu_w=np.zeros((n_obs,)),
                            use_Taylor=ssm_parameters_dict[ssm_type]["use_Taylor"])

        def fn(x):
            return f_lorenz_danse(x, device=device)
        
        def hn(x):
            return torch.from_numpy(H_).type(torch.FloatTensor).to(device) @ x
            #return torch.from_numpy(ssm_model.h_fn(x.detach().cpu().numpy())).type(torch.FloatTensor).to(device)
            
    if mode.lower() == "train": 

        #model_danse = DANSE(**estimator_options)
        model_knet = KalmanNetNN(
            n_states=estimator_options["n_states"],
            n_obs=estimator_options["n_obs"],
            n_layers=estimator_options["n_layers"],
            device=device)
        
        model_knet.Build(f=fn, h=hn)
        model_knet.ssModel = ssm_model
        tr_verbose = True
        
        # Starting model training
        tr_losses, val_losses, _ = train_KalmanNetNN(
            model=model_knet,
            options=estimator_options,
            train_loader=train_loader,
            val_loader=val_loader,
            nepochs=estimator_options["num_epochs"],
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints='some',
            device=device,
            tr_verbose=tr_verbose,
            unsupervised=estimator_options["unsupervised"]
        )
        
        #if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
            
        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        with open(os.path.join(os.path.join(logfile_path, main_exp_name), 
            '{}_gru_losses_eps{}.json'.format(knet_model_type, estimator_options["num_epochs"])), 'w') as f:
            f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":
        
        #model_danse = DANSE(**estimator_options)
        model_knet_test = KalmanNetNN(
            n_states=estimator_options["n_states"],
            n_obs=estimator_options["n_obs"],
            n_layers=estimator_options["n_layers"],
            device=device)
        model_knet.Build(f=fn, h=hn)
        model_knet.ssModel = ssm_model
        
        te_loss, _, _ = test_KalmanNetNN(
            model_test=model_knet_test,
            test_loader=test_loader,
            options=estimator_options,
            device=device,
            model_file=model_file_saved,
            test_logfile_path=te_logfile_name_with_path
        )
    
    return None

if __name__ == "__main__":
    main()
