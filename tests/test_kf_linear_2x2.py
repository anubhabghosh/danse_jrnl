#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import sys
# import tikzplotlib
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, \
    mse_loss, nmse_loss, nmse_loss_std, mse_loss_dB, mse_loss_dB_std, \
        load_saved_dataset, save_dataset, NDArrayEncoder, create_diag
#from parameters import get_parameters
from config.parameters_opt import get_parameters
from bin.ssm_models import LinearSSM
from src.kf import KF
from src.danse import DANSE, push_model
from src.k_net import KalmanNetNN
from src.dmm_causal import DMM
from parse import parse
from timeit import default_timer as timer
import json

def test_kf_linear(X, Y, kf_model):

    X_estimated_kf, Pk_estimated_kf, mse_arr_kf = kf_model.run_mb_filter(X, Y)
    return X_estimated_kf, Pk_estimated_kf, mse_arr_kf

def test_dmm_causal_linear(dmm_causal_model, saved_model_file, Y, device='cpu'):

    dmm_causal_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    dmm_causal_model = push_model(nets=dmm_causal_model, device=device)
    dmm_causal_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        _, mu_X_latent_q_seq, var_X_latent_q_seq, _, _ = dmm_causal_model.inference(Y_test_batch)
        X_estimated_filtered = mu_X_latent_q_seq
        Pk_estimated_filtered = create_diag(var_X_latent_q_seq)

    return X_estimated_filtered, Pk_estimated_filtered

def test_danse_linear(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_knet_linear(knet_model, saved_model_file, Y, device='cpu'):

    knet_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    knet_model = push_model(nets=knet_model, device=device)
    knet_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_filtered_knet = knet_model.compute_predictions(Y_test_batch)
    
    X_estimated_filtered_knet = torch.transpose(X_estimated_filtered_knet, 1, 2)
    return X_estimated_filtered_knet

def test_linear(device='cpu', model_file_saved_danse=None, model_file_saved_dmm=None, model_file_saved_knet=None, test_data_file=None, test_logfile=None, evaluation_mode=None):

    _, rnn_type, m, n, T, _, sigma_e2_dB, smnr_dB = parse("{}_danse_opt_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB", model_file_saved_danse.split('/')[-2])
    _, inference_mode_dmm, rnn_type_dmm, m, n, T, _, sigma_e2_dB, smnr_dB = parse("{}_dmm_{}_opt_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB", model_file_saved_dmm.split('/')[-2])

    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, 'a')
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        
        print('Dataset is not present, creating at {}'.format(test_data_file))
        # My own data generation scheme
        m, n, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_LinearSSM_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl", test_data_file.split('/')[-1])
        #N_test = 100 # No. of trajectories at test time / evaluation
        X = torch.zeros((N_test, T_test+1, m))
        Y = torch.zeros((N_test, T_test, n))

        # Initialize a Linear SSM with the extracted parameters
        linear_ssm = LinearSSM(n_states=m, n_obs=n,
                            mu_e=np.zeros((m,)), mu_w=np.zeros((n,)))

        print("Test data generated using sigma_e2: {} dB, SMNR: {} dB".format(sigma_e2_dB_test, smnr_dB_test))
        for i in range(N_test):
            x_lin_i, y_lin_i = linear_ssm.generate_single_sequence(T=T_test, sigma_e2_dB=sigma_e2_dB_test, smnr_dB=smnr_dB_test)
            X[i, :, :] = torch.from_numpy(x_lin_i).type(torch.FloatTensor)
            Y[i, :, :] = torch.from_numpy(y_lin_i).type(torch.FloatTensor)

        test_data_dict = {}
        test_data_dict["X"] = X
        test_data_dict["Y"] = Y
        test_data_dict["model"] = linear_ssm
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_LinearSSM_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl", test_data_file.split('/')[-1])
        test_data_dict = load_saved_dataset(filename=test_data_file)
        X = test_data_dict["X"]
        Y = test_data_dict["Y"]
        linear_ssm = test_data_dict["model"]

    print("*"*100)
    print("*"*100,file=orig_stdout)
    #i_test = np.random.choice(N_test)
    print("sigma_e2: {}dB, SMNR: {}dB".format(sigma_e2_dB_test, smnr_dB_test))
    print("sigma_e2: {}dB, SMNR: {}dB".format(sigma_e2_dB_test, smnr_dB_test), file=orig_stdout)
    #print(i_test)
    #Y = Y[:2]
    #X = X[:2]
    
    N_test, Ty, dy = Y.shape
    N_test, Tx, dx = X.shape

    # Get the estimate using the LS baseline
    H_tensor = torch.from_numpy(linear_ssm.H).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0)
    #X_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)#torch.einsum('ijj,ikj->ikj',H_tensor,Y)
    X_LS = torch.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            X_LS[i,j,:] = (torch.pinverse(H_tensor[i]) @ Y[i,j,:].reshape((dy, 1))).reshape((dx,))

    # Initialize the Kalman filter model in PyTorch
    kf_model = KF(n_states=linear_ssm.n_states,
                        n_obs=linear_ssm.n_obs,
                        F=linear_ssm.F,
                        G=linear_ssm.G,
                        H=linear_ssm.H,
                        Q=linear_ssm.Ce,
                        R=linear_ssm.Cw,
                        device=device)

    # Get the estimates using an extended Kalman filter model
    start_time_kf = timer()
    X_estimated_kf = None
    Pk_estimated_kf = None
    
    X_estimated_kf, Pk_estimated_kf, mse_arr_kf = test_kf_linear(X=X, Y=Y, kf_model=kf_model)
    time_elapsed_kf = timer() - start_time_kf

    # Initialize the DANSE model in PyTorch
    ssm_dict, est_dict = get_parameters(n_states=linear_ssm.n_states,
                                        n_obs=linear_ssm.n_obs, 
                                        device=device)

    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=linear_ssm.n_states,
        n_obs=linear_ssm.n_obs,
        mu_w=linear_ssm.mu_w,
        C_w=linear_ssm.Cw,
        batch_size=1,
        H=linear_ssm.H,
        mu_x0=np.zeros((linear_ssm.n_states,)),
        C_x0=np.eye(linear_ssm.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_filtered = None

    start_time_danse = timer()
    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_linear(danse_model=danse_model, 
                                                                                                        saved_model_file=model_file_saved_danse,
                                                                                                        Y=Y,
                                                                                                        device=device)
    
    time_elapsed_danse = timer() - start_time_danse
    
    # Initialize the KalmanNet model in PyTorch
    knet_model = KalmanNetNN(
        n_states=linear_ssm.n_states,
        n_obs=linear_ssm.n_obs,
        n_layers=1,
        device=device
    )

    def fn(x):
        return torch.from_numpy(linear_ssm.F).type(torch.FloatTensor).to(device) @ x
        
    def hn(x):
        return torch.from_numpy(linear_ssm.H).type(torch.FloatTensor).to(device) @ x
    
    knet_model.Build(f=fn, h=hn)
    knet_model.ssModel = linear_ssm

    start_time_knet = timer()

    X_estimated_filtered_knet = test_knet_linear(knet_model=knet_model, 
                                                saved_model_file=model_file_saved_knet,
                                                Y=Y,
                                                device=device)
    
    
    
    # Initialize the DMM model in PyTorch
    print("Testing DMM (ST-L)....", file=orig_stdout)
    dmm_causal_model = DMM(
        obs_dim=linear_ssm.n_obs,
        latent_dim=linear_ssm.n_states,
        rnn_model_type=est_dict['dmm']['rnn_model_type'],
        rnn_params_dict=est_dict['dmm']['rnn_params_dict'],
        optimizer_params=est_dict['dmm']['optimizer_params'],
        use_mean_field_q=est_dict['dmm']['use_mean_field_q'],
        inference_mode=est_dict['dmm']['inference_mode'],
        combiner_dim=est_dict['dmm']['combiner_dim'],
        train_emission=est_dict['dmm']['train_emission'],
        H=linear_ssm.H,
        C_w=linear_ssm.Cw,
        emission_dim=est_dict['dmm']['emission_dim'],
        emission_use_binary_obs=est_dict['dmm']['emission_use_binary_obs'],
        emission_num_layers=est_dict['dmm']['emission_num_layers'],
        train_transition=est_dict['dmm']['train_transition'],
        transition_dim=est_dict['dmm']['transition_dim'],
        transition_num_layers=est_dict['dmm']['transition_num_layers'],
        train_initials=est_dict['dmm']['train_initials'],
        device=device
    )

    X_estimated_filtered_dmm_causal = None
    Pk_estimated_filtered_dmm_causal = None

    start_time_dmm = timer()
    X_estimated_filtered_dmm_causal, Pk_estimated_filtered_dmm_causal = test_dmm_causal_linear(dmm_causal_model=dmm_causal_model, 
                                                                                            saved_model_file=model_file_saved_dmm,
                                                                                            Y=Y,
                                                                                            device=device)
    
    time_elapsed_dmm = timer() - start_time_dmm

    time_elapsed_knet = timer() - start_time_knet
    nmse_ls = nmse_loss(X[:,:,:], X_LS[:,0:,:])
    nmse_ls_std = nmse_loss_std(X[:,:,:], X_LS[:,0:,:])
    nmse_kf = nmse_loss(X[:,:,:], X_estimated_kf[:,:,:])
    nmse_kf_std = nmse_loss_std(X[:,:,:], X_estimated_kf[:,:,:])
    nmse_danse = nmse_loss(X[:,:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_std = nmse_loss_std(X[:,:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_pred = nmse_loss(X[:,:,:], X_estimated_pred[:,0:,:])
    nmse_danse_pred_std = nmse_loss_std(X[:,:,:], X_estimated_pred[:,0:,:])
    nmse_knet = nmse_loss(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    nmse_knet_std = nmse_loss_std(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    nmse_dmm = nmse_loss(X[:,:,:], X_estimated_filtered_dmm_causal[:,0:,:])
    nmse_dmm_std = nmse_loss_std(X[:,:,:], X_estimated_filtered_dmm_causal[:,0:,:])

    mse_dB_ls = mse_loss_dB(X[:,:,:], X_LS[:,0:,:])
    mse_dB_ls_std = mse_loss_dB_std(X[:,:,:], X_LS[:,0:,:])
    mse_dB_kf = mse_loss_dB(X[:,:,:], X_estimated_kf[:,:,:])
    mse_dB_kf_std = mse_loss_dB_std(X[:,:,:], X_estimated_kf[:,:,:])
    mse_dB_danse = mse_loss_dB(X[:,:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_std = mse_loss_dB_std(X[:,:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_pred = mse_loss_dB(X[:,:,:], X_estimated_pred[:,0:,:])
    mse_dB_danse_pred_std = mse_loss_dB_std(X[:,:,:], X_estimated_pred[:,0:,:])
    mse_dB_knet = mse_loss_dB(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    mse_dB_knet_std = mse_loss_dB_std(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    mse_dB_dmm = mse_loss_dB(X[:,:,:], X_estimated_filtered_dmm_causal[:,0:,:])
    mse_dB_dmm_std = mse_loss_dB_std(X[:,:,:], X_estimated_filtered_dmm_causal[:,0:,:])
    
    print("DMM CAUSAL - MSE LOSS:",mse_dB_dmm, "[dB]")
    print("DMM CAUSAL - MSE STD:", mse_dB_dmm, "[dB]")

    print("DANSE - MSE LOSS:",mse_dB_danse, "[dB]")
    print("DANSE - MSE STD:", mse_dB_danse_std, "[dB]")

    print("KNET - MSE LOSS:", mse_dB_knet, "[dB]")
    print("KNET - MSE STD:", mse_dB_knet_std, "[dB]")

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std))
    print("kf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_kf, nmse_kf_std, mse_dB_kf, mse_dB_kf_std, time_elapsed_kf))
    #print("danse, batch size: {}, nmse: {} ± {}[dB], mse: {} ± {}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    print("knet (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_knet, nmse_knet_std, mse_dB_knet, mse_dB_knet_std, time_elapsed_knet))
    print("dmm causal (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_dmm, nmse_dmm_std, mse_dB_dmm, mse_dB_dmm_std, time_elapsed_dmm))
    
    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std), file=orig_stdout)
    print("kf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_kf, nmse_kf_std, mse_dB_kf, mse_dB_kf_std, time_elapsed_kf), file=orig_stdout)
    #print("danse, batch size: {}, nmse: {} ± {}[dB], mse: {} ± {}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse), file=orig_stdout)
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse), file=orig_stdout)
    print("knet (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_knet, nmse_knet_std, mse_dB_knet, mse_dB_knet_std, time_elapsed_knet), file=orig_stdout)
    print("dmm causal (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_dmm, nmse_dmm_std, mse_dB_dmm, mse_dB_dmm_std, time_elapsed_dmm), file=orig_stdout)
    
    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X[0,:,:],0), 
                        X_est_KF=torch.squeeze(X_estimated_kf[0,:,:], 0), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[0], 0),
                        X_est_DMM=torch.squeeze(X_estimated_filtered_dmm_causal[0], 0),
                        X_est_KNET=torch.squeeze(X_estimated_filtered_knet[0], 0),
                        savefig=True,
                        savefig_name="./figs/{}/{}/AxesWisePlot_sigma_e2_{}dB_smnr_{}dB.pdf".format(dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test))
    
    plot_state_trajectory(X=torch.squeeze(X[0,:,:],0), 
                        X_est_KF=torch.squeeze(X_estimated_kf[0,:,:], 0), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[0], 0),
                        X_est_KNET=torch.squeeze(X_estimated_filtered_knet[0], 0),
                        X_est_DMM=torch.squeeze(X_estimated_filtered_dmm_causal[0], 0).numpy(), 
                        savefig=True,
                        savefig_name="./figs/{}/{}/3dPlot_sigma_e2_{}dB_smnr_{}dB.pdf".format(dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test))
    
    sys.stdout = orig_stdout
    #plt.show()
    return nmse_kf, nmse_kf_std, nmse_danse, nmse_danse_std, nmse_dmm, nmse_dmm_std, nmse_knet, nmse_knet_std, nmse_ls, nmse_ls_std, \
        mse_dB_kf, mse_dB_kf_std, mse_dB_danse, mse_dB_danse_std, mse_dB_dmm, mse_dB_dmm_std, mse_dB_knet, mse_dB_knet_std, mse_dB_ls, mse_dB_ls_std, \
        time_elapsed_kf, time_elapsed_danse, time_elapsed_dmm, time_elapsed_knet, smnr_dB_test

if __name__ == "__main__":

    # Testing parameters 
    n_states = 2
    n_obs = 2
    T_train = 100
    N_train = 500
    T_test = 1000
    N_test = 100
    sigma_e2_dB_test = -10.0
    device = 'cpu'
    evaluation_mode = 'full_opt_sigmae2_{}_N_train_{}_T_train_{}'.format(sigma_e2_dB_test, N_train, T_train)
    dirname = "LinearModel_{}x{}_w_DMM_knet".format(n_states, n_obs)
    smnr_dB_arr = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
    
    os.makedirs('./figs/{}/{}'.format(dirname, evaluation_mode), exist_ok=True)

    # Creating arrays for storing the results for post-processing and analysis
    nmse_ls_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_kf_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_dmm_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_knet_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_ls_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_kf_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_dmm_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_knet_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_kf_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_dmm_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_knet_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_kf_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_dmm_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_knet_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    t_kf_arr = np.zeros((len(smnr_dB_arr,)))
    t_danse_arr = np.zeros((len(smnr_dB_arr,)))
    t_dmm_arr = np.zeros((len(smnr_dB_arr,)))
    t_knet_arr = np.zeros((len(smnr_dB_arr,)))

    model_file_saved_dict_danse = {}
    model_file_saved_dict_dmm = {}
    model_file_saved_dict_knet = {}

    for smnr_dB in smnr_dB_arr:
        model_file_saved_dict_danse["{}dB".format(smnr_dB)] = glob.glob("./models/*LinearSSM_danse_opt*T_{}_N_{}_sigmae2_{}dB_smnr_{}dB*/*best*".format(T_train, N_train, sigma_e2_dB_test, smnr_dB))[-1]
        model_file_saved_dict_dmm["{}dB".format(smnr_dB)] = glob.glob("./models/*LinearSSM_dmm_st-l_opt*T_{}_N_{}_sigmae2_{}dB_smnr_{}dB*/*best*".format(T_train, N_train, sigma_e2_dB_test, smnr_dB))[-1]
        model_file_saved_dict_knet["{}dB".format(smnr_dB)] = glob.glob("./models/*Linear*KNetUoffline*T_{}_N_{}_sigmae2_{}dB_smnr_{}dB*/*best*".format(T_train, N_train, sigma_e2_dB_test, smnr_dB))[-1]

    print(model_file_saved_dict_danse)
    print("*"*100)
    print(model_file_saved_dict_dmm)
    print("*"*100)
    print(model_file_saved_dict_knet)
    print("*"*100)

    test_data_file_dict = {}

    for smnr_dB in smnr_dB_arr:
        test_data_file_dict["{}dB".format(smnr_dB)] = "./data/synthetic_data/test_trajectories_m_{}_n_{}_LinearSSM_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl".format(n_states, n_obs, T_test, N_test, sigma_e2_dB_test, smnr_dB)
    
    test_logfile = "./log/Linear_{}x{}_test_{}_T_{}_N_{}_sigmae2_{}_w_DMM_knet.log".format(n_states, n_obs, evaluation_mode, T_test, N_test, sigma_e2_dB_test)
    test_jsonfile = "./log/Linear_{}x{}_test_{}_T_{}_N_{}_sigmae2_{}_w_DMM_knet.json".format(n_states, n_obs, evaluation_mode, T_test, N_test, sigma_e2_dB_test)

    for i, smnr_dB in enumerate(smnr_dB_arr):
        
        model_file_saved_danse_i = model_file_saved_dict_danse['{}dB'.format(smnr_dB)]
        model_file_saved_dmm_i = model_file_saved_dict_dmm['{}dB'.format(smnr_dB)]
        test_data_file_i = test_data_file_dict['{}dB'.format(smnr_dB)]
        model_file_saved_knet_i = model_file_saved_dict_knet['{}dB'.format(smnr_dB)]

        nmse_kf_i, nmse_kf_std_i, nmse_danse_i, nmse_danse_std_i, nmse_dmm_i, nmse_dmm_std_i, nmse_knet_i, nmse_knet_std_i, nmse_ls_i, nmse_ls_std_i, \
            mse_dB_kf_i, mse_dB_kf_std_i, mse_dB_danse_i, mse_dB_danse_std_i, mse_dB_dmm_i, mse_dB_dmm_std_i, mse_dB_knet_i, mse_dB_knet_std_i, \
            mse_dB_ls_i, mse_dB_ls_std_i, time_elapsed_kf_i, time_elapsed_danse_i, time_elapsed_dmm_i, time_elapsed_knet_i, smnr_i = test_linear(device=device, 
            model_file_saved_danse=model_file_saved_danse_i, model_file_saved_dmm=model_file_saved_dmm_i, model_file_saved_knet=model_file_saved_knet_i, test_data_file=test_data_file_i, 
            test_logfile=test_logfile, evaluation_mode=evaluation_mode)
        
        # Store the NMSE values and std devs of the NMSE values
        nmse_ls_arr[i] = nmse_ls_i.numpy().item()
        nmse_kf_arr[i] = nmse_kf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.numpy().item()
        nmse_dmm_arr[i] = nmse_dmm_i.numpy().item()
        nmse_knet_arr[i] = nmse_knet_i.numpy().item()
        nmse_ls_std_arr[i] = nmse_ls_std_i.numpy().item()
        nmse_kf_std_arr[i] = nmse_kf_std_i.numpy().item()
        nmse_danse_std_arr[i] = nmse_danse_std_i.numpy().item()
        nmse_dmm_std_arr[i] = nmse_dmm_std_i.numpy().item()
        nmse_knet_std_arr[i] = nmse_knet_std_i.numpy().item()

        # Store the MSE values and std devs of the MSE values (in dB)
        mse_ls_dB_arr[i] = mse_dB_ls_i.numpy().item()
        mse_kf_dB_arr[i] = mse_dB_kf_i.numpy().item()
        mse_danse_dB_arr[i] = mse_dB_danse_i.numpy().item()
        mse_dmm_dB_arr[i] = mse_dB_dmm_i.numpy().item()
        mse_knet_dB_arr[i] = mse_dB_knet_i.numpy().item()
        mse_ls_dB_std_arr[i] = mse_dB_ls_std_i.numpy().item()
        mse_kf_dB_std_arr[i] = mse_dB_kf_std_i.numpy().item()
        mse_danse_dB_std_arr[i] = mse_dB_danse_std_i.numpy().item()
        mse_dmm_dB_std_arr[i] = mse_dB_dmm_std_i.numpy().item()
        mse_knet_dB_std_arr[i] = mse_dB_knet_std_i.numpy().item()

        # Store the inference times 
        t_kf_arr[i] = time_elapsed_kf_i
        t_danse_arr[i] = time_elapsed_danse_i
        t_dmm_arr[i] = time_elapsed_dmm_i
        t_knet_arr[i] = time_elapsed_knet_i

    # Collect stats in a large json file
    test_stats = {}
    test_stats['KF_mean_nmse'] = nmse_kf_arr
    test_stats['DANSE_mean_nmse'] = nmse_danse_arr
    test_stats['DMM_mean_nmse'] = nmse_dmm_arr
    test_stats['KNET_mean_nmse'] = nmse_knet_arr
    test_stats['KF_std_nmse'] = nmse_kf_std_arr
    test_stats['DANSE_std_nmse'] = nmse_danse_std_arr
    test_stats['DMM_std_nmse'] = nmse_dmm_std_arr
    test_stats['KNET_std_nmse'] = nmse_knet_std_arr
    test_stats['KF_mean_mse'] = mse_kf_dB_arr
    test_stats['DANSE_mean_mse'] = mse_danse_dB_arr
    test_stats['DMM_mean_mse'] = mse_dmm_dB_arr
    test_stats['KNET_mean_mse'] = mse_knet_dB_arr
    test_stats['KF_std_mse'] = mse_kf_dB_std_arr
    test_stats['DANSE_std_mse'] = mse_danse_dB_std_arr
    test_stats['DMM_std_mse'] = mse_dmm_dB_std_arr
    test_stats['KNET_std_mse'] = mse_knet_dB_std_arr
    test_stats['LS_mean_mse'] = mse_ls_dB_arr
    test_stats['LS_std_mse'] = mse_ls_dB_std_arr
    test_stats['LS_mean_nmse'] = nmse_ls_arr
    test_stats['LS_std_nmse'] = nmse_ls_std_arr
    test_stats['KF_time'] = t_kf_arr
    test_stats['DANSE_time'] = t_danse_arr
    test_stats['DMM_time'] = t_dmm_arr
    test_stats['KNET_time'] = t_knet_arr
    test_stats['SMNR'] = smnr_dB_arr
    
    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(test_stats, cls=NDArrayEncoder, indent=2))

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    # Plotting figures for later use
    plt.figure()
    plt.errorbar(smnr_dB_arr, nmse_ls_arr, fmt='gp-.', yerr=nmse_ls_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, nmse_kf_arr, fmt='rd--',  yerr=nmse_kf_std_arr, linewidth=1.5, label="KF")
    plt.errorbar(smnr_dB_arr, nmse_danse_arr, fmt='b*-', yerr=nmse_danse_std_arr, linewidth=2.0, label="DANSE")
    plt.errorbar(smnr_dB_arr, nmse_dmm_arr, fmt='y*-', yerr=nmse_dmm_std_arr, linewidth=2.0, label="DMM (ST-L)")
    plt.errorbar(smnr_dB_arr, nmse_knet_arr, fmt='ys-', yerr=nmse_knet_std_arr,  linewidth=1.0, label="KalmanNet")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/{}/{}/NMSE_vs_SMNR_Linear_2x2.pdf'.format(dirname, evaluation_mode))
    # tikzplotlib.save('./figs/{}/{}/NMSE_vs_SMNR_Linear_2x2.tex'.format(dirname, evaluation_mode))
    
    plt.figure()
    plt.errorbar(smnr_dB_arr, mse_ls_dB_arr, fmt='gp-.', yerr=mse_ls_dB_std_arr, linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, mse_kf_dB_arr, fmt='rd--',  yerr=mse_kf_dB_std_arr, linewidth=1.5, label="KF")
    plt.errorbar(smnr_dB_arr, mse_danse_dB_arr, fmt='b*-', yerr=mse_danse_dB_std_arr, linewidth=2.0, label="DANSE")
    plt.errorbar(smnr_dB_arr, mse_dmm_dB_arr, fmt='y*-', yerr=mse_dmm_dB_std_arr, linewidth=2.0, label="DMM (ST-L)")
    plt.errorbar(smnr_dB_arr, mse_knet_dB_arr, fmt='ys-', yerr=mse_knet_dB_std_arr, linewidth=1.0, label="KalmanNet")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/{}/{}/MSE_vs_SMNR_Linear_2x2.pdf'.format(dirname, evaluation_mode))
    # tikzplotlib.save('./figs/{}/{}/MSE_vs_SMNR_Linear_2x2.tex'.format(dirname, evaluation_mode))

    #plt.subplot(212)
    plt.figure()
    plt.plot(smnr_dB_arr, t_kf_arr, 'rd--',  linewidth=1.5, label="KF")
    plt.plot(smnr_dB_arr, t_danse_arr, 'b*-', linewidth=2.0, label="DANSE")
    plt.plot(smnr_dB_arr, t_dmm_arr, 'b*-', linewidth=2.0, label="DMM (ST-L)")
    plt.plot(smnr_dB_arr, t_knet_arr, 'ys-', linewidth=1.0, label="KalmanNet")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('Inference time (in s)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/{}/{}/InferTime_vs_SMNR_Linear_2x2.pdf'.format(dirname, evaluation_mode))
    # tikzplotlib.save('./figs/{}/{}/InferTime_vs_SMNR_Linear_2x2.tex'.format(dirname, evaluation_mode))
    
    #plt.show()
