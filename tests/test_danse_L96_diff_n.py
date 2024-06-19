#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
import numpy as np
import glob
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse
from timeit import default_timer as timer
import json
# import tikzplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, \
    mse_loss_dB, load_saved_dataset, save_dataset, nmse_loss_std, mse_loss_dB_std, NDArrayEncoder, partial_corrupt
#from parameters import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test
from config.parameters_opt import get_parameters, A_fn, h_fn, delta_t_L96, get_H_DANSE
from bin.ssm_models import LorenzSSM, Lorenz96SSM
from src.ekf import EKF
from src.ukf import UKF
from src.ukf_aliter import UKF_Aliter
from src.k_net import KalmanNetNN
from src.danse import DANSE, push_model

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
def test_danse_lorenz(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_lorenz(device='cpu', model_file_saved=None, model_file_saved_knet=None, test_data_file=None, test_logfile=None, evaluation_mode='Full', p=0.5, bias=30):

    dataset_type, rnn_type, m, n, T, _, sigma_e2_dB, smnr_dB = parse("{}_danse_opt_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB", model_file_saved.split('/')[-2])
    delta = delta_t_L96 # If decimate is True, then set this delta to 1e-5 and run it for long time
    #smnr_dB = 20
    decimate=False
    
    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, 'a')
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        
        print('Dataset is not present, creating at {}'.format(test_data_file))
        # My own data generation scheme
        m, n, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_Lorenz96SSM_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl", test_data_file.split('/')[-1])
        #N_test = 100 # No. of trajectories at test time / evaluation
        X = torch.zeros((N_test, T_test+1, m))
        Y = torch.zeros((N_test, T_test, n))

        lorenz_model = Lorenz96SSM(
            n_states=m, n_obs=n, delta=delta, delta_d=delta/2, F_mu=8.0, decimate=decimate, 
            mu_w=np.zeros((n,)), method='RK45', H=get_H_DANSE(type_=dataset_type, n_states=m, n_obs=n)
        )

        print("Test data generated using sigma_e2: {} dB, SMNR: {} dB".format(sigma_e2_dB_test, smnr_dB_test))
        
        for i in range(N_test):
            x_lorenz_i, y_lorenz_i = lorenz_model.generate_single_sequence(T=T_test, sigma_e2_dB=sigma_e2_dB_test, smnr_dB=smnr_dB_test)
            X[i, :, :] = torch.from_numpy(x_lorenz_i).type(torch.FloatTensor)
            Y[i, :, :] = torch.from_numpy(y_lorenz_i).type(torch.FloatTensor)

        test_data_dict = {}
        test_data_dict["X"] = X
        test_data_dict["Y"] = Y
        test_data_dict["model"] = lorenz_model
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_Lorenz96SSM_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl", test_data_file.split('/')[-1])
        test_data_dict = load_saved_dataset(filename=test_data_file)
        X = test_data_dict["X"]
        Y = test_data_dict["Y"]
        lorenz_model = test_data_dict["model"]

    print("*"*100)
    print("*"*100,file=orig_stdout)
    #i_test = np.random.choice(N_test)
    print("sigma_e2: {}dB, smnr: {}dB, n_obs: {}".format(sigma_e2_dB_test, smnr_dB_test, n))
    print("sigma_e2: {}dB, smnr: {}dB, n_obs: {}".format(sigma_e2_dB_test, smnr_dB_test, n), file=orig_stdout)

    #Y = Y[:2]
    #X = X[:2]

    N_test, Ty, dy = Y.shape
    N_test, Tx, dx = X.shape

    # Get the estimate using the baseline
    print(lorenz_model.H.shape)
    H_tensor = torch.from_numpy(lorenz_model.H).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0)
    #X_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)
    X_LS = torch.zeros((N_test, Ty, dx))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            print(torch.pinverse(H_tensor[i]).shape)
            X_LS[i,j,:] = (torch.pinverse(H_tensor[i]) @ Y[i,j,:].reshape((dy, 1))).reshape((dx,))

    if "partial" in evaluation_mode:
        if "low" in evaluation_mode:
            p *= -1
            bias *= -1
        elif "high" in evaluation_mode:
            p *= 1
            bias *= 1

        sigma_e2_dB_test = partial_corrupt((sigma_e2_dB_test), p=p, bias=bias) # 50 % corruption of the true nu_dB used for data generation
    
    # NOTE: Partial corruption code is incomplete! Needs to be fixed !!

    #print("Fed to Model-based filters: ", file=orig_stdout)
    #print("sigma_e2: {}dB, smnr: {}dB, delta_t: {}".format(sigma_e2_dB_test, smnr_dB_test, delta_t_L96), file=orig_stdout)

    #print("Fed to Model-based filters: ")
    #print("sigma_e2: {}dB, smnr: {}dB, delta_t: {}".format(sigma_e2_dB_test, smnr_dB_test, delta_t_L96))

    lorenz_model.sigma_e2 = dB_to_lin(sigma_e2_dB_test)
    lorenz_model.setStateCov(sigma_e2=dB_to_lin(sigma_e2_dB_test))

    print("Testing DANSE ...", file=orig_stdout)
    # Initialize the DANSE model in PyTorch
    ssm_dict, est_dict = get_parameters(n_states=lorenz_model.n_states,
                                        n_obs=lorenz_model.n_obs, 
                                        device=device)
    
    
    time_elapsed_ukf =  None #timer() - start_time_ukf
    time_elapsed_ekf = None #timer() - start_time_ukf
    
    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        mu_w=lorenz_model.mu_w,
        C_w=lorenz_model.Cw,
        batch_size=1,
        H=lorenz_model.H,#jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy(),
        mu_x0=np.zeros((lorenz_model.n_states,)),
        C_x0=np.eye(lorenz_model.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )
    
    print("DANSE Model file: {}".format(model_file_saved))

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_filtered = None

    start_time_danse = timer()
    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_lorenz(danse_model=danse_model, 
                                                                                                saved_model_file=model_file_saved,
                                                                                                Y=Y,
                                                                                                device=device)
    time_elapsed_danse = timer() - start_time_danse

    '''
    print("Testing KalmanNet ...", file=orig_stdout)
    # Initialize the KalmanNet model in PyTorch
    knet_model = KalmanNetNN(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        n_layers=1,
        device=device
    )

    def fn(x):
        return f_lorenz_danse_knet(x, device=device)
        
    def hn(x):
        #return x
        return lorenz_model.h_fn(x)
    
    knet_model.Build(f=fn, h=hn)
    knet_model.ssModel = lorenz_model

    start_time_knet = timer()

    X_estimated_filtered_knet = test_knet_lorenz(knet_model=knet_model, 
                                                saved_model_file=model_file_saved_knet,
                                                Y=Y,
                                                device=device)
    
    time_elapsed_knet = timer() - start_time_knet
    '''
    time_elapsed_knet = None
    nmse_ls = nmse_loss(X[:,:,:], X_LS[:,0:,:])
    nmse_ls_std = nmse_loss_std(X[:,:,:], X_LS[:,0:,:])
    nmse_ekf = None #nmse_loss(X[:,:,:], X_estimated_ekf[:,:,:])
    nmse_ekf_std = None # nmse_loss_std(X[:,:,:], X_estimated_ekf[:,:,:])
    nmse_ukf = None #nmse_loss(X[:,:,:], X_estimated_ukf[:,:,:])
    nmse_ukf_std = None #nmse_loss_std(X[:,:,:], X_estimated_ukf[:,:,:])
    nmse_danse = nmse_loss(X[:,:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_std = nmse_loss_std(X[:,:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_pred = nmse_loss(X[:,:,:], X_estimated_pred[:,0:,:])
    nmse_danse_pred_std = nmse_loss_std(X[:,:,:], X_estimated_pred[:,0:,:])
    nmse_knet = None #nmse_loss(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    nmse_knet_std = None #nmse_loss_std(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    
    mse_dB_ls = mse_loss_dB(X[:,:,:], X_LS[:,0:,:])
    mse_dB_ls_std = mse_loss_dB_std(X[:,:,:], X_LS[:,0:,:])
    mse_dB_ekf = None #mse_loss_dB(X[:,:,:], X_estimated_ekf[:,:,:])
    mse_dB_ekf_std = None #mse_loss_dB_std(X[:,:,:], X_estimated_ekf[:,:,:])
    mse_dB_ukf = None #mse_loss_dB(X[:,:,:], X_estimated_ukf[:,:,:])
    mse_dB_ukf_std = None #mse_loss_dB_std(X[:,:,:], X_estimated_ukf[:,:,:])
    mse_dB_danse = mse_loss_dB(X[:,:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_std = mse_loss_dB_std(X[:,:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_pred = mse_loss_dB(X[:,:,:], X_estimated_pred[:,0:,:])
    mse_dB_danse_pred_std = mse_loss_dB_std(X[:,:,:], X_estimated_pred[:,0:,:])
    mse_dB_knet = None #mse_loss_dB(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    mse_dB_knet_std = None #mse_loss_dB_std(X[:,:,:], X_estimated_filtered_knet[:,0:,:])
    
    print("DANSE - MSE LOSS:",mse_dB_danse, "[dB]")
    print("DANSE - MSE STD:", mse_dB_danse_std, "[dB]")

    print("KNET - MSE LOSS:", mse_dB_knet, "[dB]")
    print("KNET - MSE STD:", mse_dB_knet_std, "[dB]")

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std))
    #print("ekf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_ekf, nmse_ekf_std, mse_dB_ekf, mse_dB_ekf_std, time_elapsed_ekf))
    #print("ukf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_ukf, nmse_ukf_std, mse_dB_ukf, mse_dB_ukf_std, time_elapsed_ukf))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    #print("knet (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_knet, nmse_knet_std, mse_dB_knet, mse_dB_knet_std, time_elapsed_knet))

    # System console print
    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std), file=orig_stdout)
    #print("ekf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_ekf, nmse_ekf_std, mse_dB_ekf, mse_dB_ekf_std, time_elapsed_ekf), file=orig_stdout)
    #print("ukf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_ukf, nmse_ukf_std, mse_dB_ukf, mse_dB_ukf_std, time_elapsed_ukf), file=orig_stdout)
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse), file=orig_stdout)
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse), file=orig_stdout)
    #print("knet (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_knet, nmse_knet_std, mse_dB_knet, mse_dB_knet_std, time_elapsed_knet), file=orig_stdout)

    # Plot the result
    j = np.random.randint(X.shape[0])
    plot_state_trajectory(X=torch.squeeze(X[j,:,:],0).numpy(), 
                        #X_est_EKF=torch.squeeze(X_estimated_ekf[0,:,:],0).numpy(), 
                        #X_est_UKF=torch.squeeze(X_estimated_ukf[0,:,:],0).numpy(), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[j],0).numpy(),
                        #X_est_KNET=torch.squeeze(X_estimated_filtered_knet[0], 0).numpy(),
                        savefig=True,
                        savefig_name="./figs/Lorenz96Model/{}/3dPlot_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n))
    plt.close()
    plot_multidim_imshow(X=torch.squeeze(X[j,:,:],0).numpy(), X_est_DANSE=torch.squeeze(X_estimated_filtered[j],0).numpy(), 
                         Y=torch.squeeze(Y[j,:,:],0).numpy(),
                         savefig=True, 
                         savefig_name="./figs/Lorenz96Model/{}/Multidim_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n))
    plt.close()
    plot_3d_state_trajectory(X=torch.squeeze(X[j, :, :], 0).numpy(), legend='$\\mathbf{x}^{true}$', m='b-', savefig_name="./figs/Lorenz96Model/{}/lorenz96ssm_x_true_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n), savefig=True)
    plt.close()
    plot_3d_state_trajectory(X=torch.squeeze(X_estimated_filtered[j], 0).numpy(), legend='$\\hat{\mathbf{x}}_{DANSE}$', m='k-', savefig_name="./figs/Lorenz96Model/{}/lorenz96ssm_x_danse_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n), savefig=True)
    plt.close()
    plot_3d_measurment_trajectory(Y=torch.squeeze(Y[j, :, :], 0).numpy(), legend='$\\mathbf{y}^{true}$', m='r-', savefig_name="./figs/Lorenz96Model/{}/lorenz96ssm_y_true_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n), savefig=True)
    plt.close()
    plot_state_trajectory_w_lims(X=torch.squeeze(X[j,:,:],0).numpy(), 
                        #X_est_KF=torch.squeeze(X_estimated_kf[0,:,:], 0).numpy(), 
                        #X_est_KF_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_kf[0,:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()), 
                        #X_est_UKF=torch.squeeze(X_estimated_ukf[j,:,:], 0).numpy(), 
                        #X_est_UKF_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_ukf[j,:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[j], 0).numpy(), 
                        X_est_DANSE_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_filtered[j], 0), offset=0, dim1=1,dim2=2).numpy()), 
                        #X_est_DANSE_sup=torch.squeeze(X_estimated_filtered_sup[0], 0).numpy(), 
                        #X_est_DANSE_sup_std=np.diag(torch.squeeze(Pk_estimated_filtered_sup[0,:,:], 0).numpy()).sqrt(), 
                        #X_est_KNET=torch.squeeze(X_estimated_filtered_knet[0], 0).numpy(), 
                        savefig=True,
                        savefig_name="./figs/Lorenz96Model/{}/Trajectories_sigma_e2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n))
    plt.close()
    plot_state_trajectory_axes(X=torch.squeeze(X[j,:,:],0).numpy(), 
                                #X_est_EKF=torch.squeeze(X_estimated_ekf[0,:,:],0).numpy(), 
                                #X_est_UKF=torch.squeeze(X_estimated_ukf[j,:,:],0).numpy(), 
                                X_est_DANSE=torch.squeeze(X_estimated_filtered[j],0).numpy(), 
                                #X_est_KNET=torch.squeeze(X_estimated_filtered_knet[0], 0).numpy(),
                                savefig=True,
                                savefig_name="./figs/Lorenz96Model/{}/AxesWisePlot_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n))
    plt.close()
    plot_state_trajectory_axes_all(X=torch.squeeze(X[j,:,:],0).numpy(), 
                                #X_est_EKF=torch.squeeze(X_estimated_ekf[0,:,:],0).numpy(), 
                                #X_est_UKF=torch.squeeze(X_estimated_ukf[j,:,:],0).numpy(), 
                                X_est_DANSE=torch.squeeze(X_estimated_filtered[j],0).numpy(), 
                                #X_est_KNET=torch.squeeze(X_estimated_filtered_knet[0], 0).numpy(),
                                savefig=True,
                                savefig_name="./figs/Lorenz96Model/{}/AxesAllPlot_sigmae2_{}dB_smnr_{}dB_nobs_{}.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test, n))
    plt.close()
    #plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    #plot_state_trajectory(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    
    #plt.show()
    sys.stdout = orig_stdout
    return nmse_ekf, nmse_ekf_std, nmse_danse, nmse_danse_std, nmse_knet, nmse_knet_std, nmse_ukf, nmse_ukf_std, nmse_ls, nmse_ls_std, \
        mse_dB_ekf, mse_dB_ekf_std, mse_dB_danse, mse_dB_danse_std, mse_dB_knet, mse_dB_knet_std, mse_dB_ukf, mse_dB_ukf_std, mse_dB_ls, mse_dB_ls_std, \
        time_elapsed_ekf, time_elapsed_danse, time_elapsed_knet, time_elapsed_ukf, smnr_dB_test

if __name__ == "__main__":

    # Testing parameters 
    T_test = 2000
    N_test = 100
    sigma_e2_dB_test = -10.0
    smnr_dB_test = 10.0
    n_states = 20
    device = 'cpu'
    bias = None # By default should be positive, equal to 10.0
    p = None # Keep this fixed at zero for now, equal to 0.0
    mode = 'full'
    if mode == 'low' or mode == 'high':
        evaluation_mode = 'partial_opt_{}_bias_{}_p_{}'.format(mode, bias, p)
    else:
        bias = None
        p = None
        evaluation_mode = 'full_opt_bias_{}_p_{}'.format(None, None)

    os.makedirs('./figs/Lorenz96Model/{}'.format(evaluation_mode), exist_ok=True)

    n_obs_arr = np.array([20]) #np.array([4,7,10,13,15,19,20])

    nmse_ls_arr = np.zeros((len(n_obs_arr,)))
    nmse_ls_arr = np.zeros((len(n_obs_arr,)))
    nmse_ekf_arr = np.zeros((len(n_obs_arr,)))
    nmse_ukf_arr = np.zeros((len(n_obs_arr,)))
    nmse_danse_arr = np.zeros((len(n_obs_arr,)))
    nmse_knet_arr = np.zeros((len(n_obs_arr,)))
    nmse_ls_std_arr = np.zeros((len(n_obs_arr,)))
    nmse_ekf_std_arr = np.zeros((len(n_obs_arr,)))
    nmse_ukf_std_arr = np.zeros((len(n_obs_arr,)))
    nmse_danse_std_arr = np.zeros((len(n_obs_arr,)))
    nmse_knet_std_arr = np.zeros((len(n_obs_arr,)))
    mse_ls_dB_arr = np.zeros((len(n_obs_arr,)))
    mse_ekf_dB_arr = np.zeros((len(n_obs_arr,)))
    mse_ukf_dB_arr = np.zeros((len(n_obs_arr,)))
    mse_danse_dB_arr = np.zeros((len(n_obs_arr,)))
    mse_knet_dB_arr = np.zeros((len(n_obs_arr,)))
    mse_ls_dB_std_arr = np.zeros((len(n_obs_arr,)))
    mse_ekf_dB_std_arr = np.zeros((len(n_obs_arr,)))
    mse_ukf_dB_std_arr = np.zeros((len(n_obs_arr,)))
    mse_danse_dB_std_arr = np.zeros((len(n_obs_arr,)))
    mse_knet_dB_std_arr = np.zeros((len(n_obs_arr,)))
    t_ekf_arr = np.zeros((len(n_obs_arr,)))
    t_ukf_arr = np.zeros((len(n_obs_arr,)))
    t_danse_arr = np.zeros((len(n_obs_arr,)))
    t_knet_arr = np.zeros((len(n_obs_arr,)))
    snr_arr = np.zeros((len(n_obs_arr,)))

    model_file_saved_dict = {}
    model_file_saved_dict_knet = {}

    sigma_e2_dB_nominal = -10.0 

    for n_obs in n_obs_arr:

        if n_obs < n_states:
            model_file_saved_dict["{}".format(n_obs)] = glob.glob("./models/*Lorenz96SSMn{}_danse_opt_*n_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(n_obs, n_obs, sigma_e2_dB_nominal, smnr_dB_test))[-1]
        else:
            model_file_saved_dict["{}".format(n_obs)] = glob.glob("./models/*Lorenz96SSM_danse_opt_*n_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(n_obs, sigma_e2_dB_nominal, smnr_dB_test))[-1]
        model_file_saved_dict_knet["{}".format(n_obs)] = None #glob.glob("./models/*Chen*KNetUoffline_*n_3*sigmae2_{}dB_smnr_{}dB*/*best*".format(sigma_e2_dB_test, smnr_dB))[-1]

    test_data_file_dict = {}

    for n_obs in n_obs_arr:
        test_data_file_dict["{}".format(n_obs)] = "./data/synthetic_data/test_trajectories_m_{}_n_{}_Lorenz96SSM_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl".format(n_states, n_obs, T_test, N_test, sigma_e2_dB_test, smnr_dB_test)
    
    print(model_file_saved_dict)

    test_logfile = "./log/Lorenz96SSM_test_{}_T_{}_N_{}_w_knet.log".format(evaluation_mode, T_test, N_test)
    test_jsonfile = "./log/Lorenz96SSM_test_{}_T_{}_N_{}_w_knet.json".format(evaluation_mode, T_test, N_test)

    for i, n_obs in enumerate(n_obs_arr):
        
        model_file_saved_i = model_file_saved_dict['{}'.format(n_obs)]
        test_data_file_i = test_data_file_dict['{}'.format(n_obs)]
        model_file_saved_knet_i = model_file_saved_dict_knet['{}'.format(n_obs)]

        nmse_ekf_i, nmse_ekf_i_std, nmse_danse_i, nmse_danse_i_std, nmse_knet_i, nmse_knet_std_i, nmse_ukf_i, nmse_ukf_i_std, nmse_ls_i, nmse_ls_i_std, \
            mse_dB_ekf_i, mse_dB_ekf_std_i, mse_dB_danse_i, mse_dB_danse_std_i, mse_dB_knet_i, mse_dB_knet_std_i, mse_dB_ukf_i, mse_dB_ukf_std_i, mse_dB_ls_i, mse_dB_ls_std_i, \
            time_elapsed_ekf_i, time_elapsed_danse_i, time_elapsed_knet_i, time_elapsed_ukf_i, smnr_dB_i = test_lorenz(device=device, 
            model_file_saved=model_file_saved_i, model_file_saved_knet=model_file_saved_knet_i, test_data_file=test_data_file_i, test_logfile=test_logfile, 
            evaluation_mode=evaluation_mode, bias=bias, p=p)

        
        # Store the NMSE values and std devs of the NMSE values
        #nmse_ls_arr[i] = nmse_ls_i.numpy().item()
        #nmse_ekf_arr[i] = nmse_ekf_i.numpy().item()
        #nmse_ukf_arr[i] = nmse_ukf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.numpy().item()
        #nmse_knet_arr[i] = nmse_knet_i.numpy().item()
        #nmse_ls_std_arr[i] = nmse_ls_i_std.numpy().item()
        #nmse_ekf_std_arr[i] = nmse_ekf_i_std.numpy().item()
        #nmse_ukf_std_arr[i] = nmse_ukf_i_std.numpy().item()
        nmse_danse_std_arr[i] = nmse_danse_i_std.numpy().item()
        #nmse_knet_std_arr[i] = nmse_knet_std_i.numpy().item()
        
        # Store the MSE values and std devs of the MSE values (in dB)
        #mse_ls_dB_arr[i] = mse_dB_ls_i.numpy().item()
        #mse_ekf_dB_arr[i] = mse_dB_ekf_i.numpy().item()
        #mse_ukf_dB_arr[i] = mse_dB_ukf_i.numpy().item()
        mse_danse_dB_arr[i] = mse_dB_danse_i.numpy().item()
        #mse_knet_dB_arr[i] = mse_dB_knet_i.numpy().item()
        #mse_ls_dB_std_arr[i] = mse_dB_ls_std_i.numpy().item()
        #mse_ekf_dB_std_arr[i] = mse_dB_ekf_std_i.numpy().item()
        #mse_ukf_dB_std_arr[i] = mse_dB_ukf_std_i.numpy().item()
        mse_danse_dB_std_arr[i] = mse_dB_danse_std_i.numpy().item()
        #mse_knet_dB_std_arr[i] = mse_dB_knet_std_i.numpy().item()

        # Store the inference times
        #t_ekf_arr[i] = time_elapsed_ekf_i
        #t_ukf_arr[i] = time_elapsed_ukf_i
        t_danse_arr[i] = time_elapsed_danse_i
        #t_knet_arr[i] = time_elapsed_knet_i
        
    
    test_stats = {}
    #test_stats['UKF_mean_nmse'] = nmse_ukf_arr
    #test_stats['EKF_mean_nmse'] = nmse_ekf_arr
    test_stats['DANSE_mean_nmse'] = nmse_danse_arr
    #test_stats['KNET_mean_nmse'] = nmse_knet_arr
    #test_stats['UKF_std_nmse'] = nmse_ukf_std_arr
    #test_stats['EKF_std_nmse'] = nmse_ekf_std_arr
    test_stats['DANSE_std_nmse'] = nmse_danse_std_arr
    #test_stats['KNET_std_nmse'] = nmse_knet_std_arr
    #test_stats['LS_mean_nmse'] = nmse_ls_arr
    #test_stats['LS_std_nmse'] = nmse_ls_std_arr

    #test_stats['EKF_mean_mse'] = mse_ekf_dB_arr
    #test_stats['UKF_mean_mse'] = mse_ukf_dB_arr
    test_stats['DANSE_mean_mse'] = mse_danse_dB_arr
    #test_stats['KNET_mean_mse'] = mse_knet_dB_arr
    #test_stats['EKF_std_mse'] = mse_ekf_dB_std_arr
    #test_stats['UKF_std_mse'] = mse_ukf_dB_std_arr
    test_stats['DANSE_std_mse'] = mse_danse_dB_std_arr
    #test_stats['KNET_std_mse'] = mse_knet_dB_std_arr
    #test_stats['LS_mean_mse'] = mse_ls_dB_arr
    #test_stats['LS_std_mse'] = mse_ls_dB_std_arr

    #test_stats['UKF_time'] = t_ukf_arr
    #test_stats['EKF_time'] = t_ekf_arr
    test_stats['DANSE_time'] = t_danse_arr
    #test_stats['KNET_time'] = t_knet_arr
    test_stats['n_obs'] = n_obs_arr
    
    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(test_stats, cls=NDArrayEncoder, indent=2))

    # Plotting the NMSE Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    #plt.errorbar(n_obs_arr, nmse_ls_arr, fmt='gp-.', yerr=nmse_ls_std_arr,  linewidth=1.5, label="LS")
    #plt.errorbar(n_obs_arr, nmse_ekf_arr, fmt='rd--',  yerr=nmse_ekf_std_arr, linewidth=1.5, label="EKF")
    #plt.errorbar(n_obs_arr, nmse_ukf_arr, fmt='ko-',  yerr=nmse_ukf_std_arr, linewidth=1.5, label="UKF")
    plt.errorbar(n_obs_arr / n_states, nmse_danse_arr, fmt='b*-', yerr=nmse_danse_std_arr, linewidth=2.0, label="DANSE")
    #plt.errorbar(n_obs_arr, nmse_knet_arr, fmt='ys-', yerr=nmse_knet_std_arr,  linewidth=1.0, label="KalmanNet")
    plt.xlabel('n / m')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    # tikzplotlib.save('./figs/Lorenz96Model/{}/NMSE_vs_n_Lorenz.tex'.format(evaluation_mode))
    plt.savefig('./figs/Lorenz96Model/{}/NMSE_vs_n_Lorenz.pdf'.format(evaluation_mode))

    # Plotting the Time-elapsed Curve
    plt.figure()
    #plt.subplot(211)
    #plt.plot(n_obs_arr, t_ekf_arr, 'rd--', linewidth=1.5, label="EKF")
    #plt.plot(n_obs_arr, t_ukf_arr, 'ks--', linewidth=1.5, label="UKF")
    plt.plot(n_obs_arr / n_states, t_danse_arr, 'bo-', linewidth=2.0, label="DANSE")
    #plt.plot(n_obs_arr, t_knet_arr, 'ys-', linewidth=1.0, label="KalmanNet")
    plt.xlabel('n / m')
    plt.ylabel('Inference time (in s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # tikzplotlib.save('./figs/Lorenz96Model/{}/InferTime_vs_n_Lorenz_w_knet.tex'.format(evaluation_mode))
    plt.savefig('./figs/Lorenz96Model/{}/InferTime_vs_n_Lorenz_w_knet.pdf'.format(evaluation_mode))

    # Plotting the MSE Curve
    plt.figure()
    #plt.errorbar(n_obs_arr, mse_ls_dB_arr, fmt='gp-.', yerr=mse_ls_dB_std_arr,  linewidth=1.5, label="LS")
    #plt.errorbar(n_obs_arr, mse_ekf_dB_arr, fmt='rd--',  yerr=mse_ekf_dB_std_arr, linewidth=1.5, label="EKF")
    #plt.errorbar(n_obs_arr, mse_ukf_dB_arr, fmt='ko-',  yerr=mse_ukf_dB_std_arr, linewidth=1.5, label="UKF")
    plt.errorbar(n_obs_arr / n_states, mse_danse_dB_arr, fmt='b*-', yerr=mse_danse_dB_std_arr, linewidth=2.0, label="DANSE")
    #plt.errorbar(n_obs_arr, mse_knet_dB_arr, fmt='ys-', yerr=mse_knet_dB_std_arr,  linewidth=1.0, label="KalmanNet")
    plt.xlabel('n / m')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    # tikzplotlib.save('./figs/Lorenz96Model/{}/MSE_vs_n_Lorenz_w_knet.tex'.format(evaluation_mode))
    plt.savefig('./figs/Lorenz96Model/{}/MSE_vs_n_Lorenz_w_knet.pdf'.format(evaluation_mode))
    
    #plt.show()
