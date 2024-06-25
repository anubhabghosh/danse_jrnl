#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
# Adopted from: https://github.com/KalmanNet/KalmanNet_TSP 
#####################################################
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints 
import torch
from torch import nn
from timeit import default_timer as timer
from utils.utils import dB_to_lin, mse_loss
from scipy.linalg import expm
#from parameters import delta_t, J_test
import numpy as np
import math

def A_fn_exact(z, dt):
    return expm(np.array([
                    [-10, 10, 0],
                    [28, -1, -z[0]],
                    [0, z[0], -8.0/3]
                ])*dt) @ z

'''
def f_lorenz_danse(x, dt):

    x = torch.from_numpy(x).type(torch.FloatTensor)
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) 
    #A = torch.reshape(torch.matmul(B, x),(3,3)).T # For KalmanNet
    A += C
    #delta = delta_t # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()
'''

class UKF_Aliter(nn.Module):
    """ This class implements an unscented Kalman filter in PyTorch
    """
    def __init__(self, n_states, n_obs, f=None, h=None, Q=None, R=None, kappa=-1, 
                alpha=0.1, beta=2, delta_t=1.0, device='cpu', init_cond=None):
        super(UKF_Aliter, self).__init__()

        # Initialize the device
        self.device = device

        # Initializing the system model
        self.n_states = n_states # Setting the number of states of the Kalman filter
        self.n_obs = n_obs
        self.f_k = f # State transition function (relates x_k, u_k to x_{k+1})
        self.h_k = h # Output function (relates state x_k to output y_k)
         

        self.Q_k = self.push_to_device(Q) # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = self.push_to_device(R) # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)
        
        # Defining the parameters for the unscented transformation
        self.kappa = kappa # Hyperparameter in the UT
        self.alpha = alpha # Hyperparameter in the UT
        self.beta = beta # Hyperparameter in the UT

        self.get_sigma_points()
        #self.set_mean_weights()
        #self.set_cov_weights()

        #self.init_cond = init_cond

        self.delta_t = delta_t
        self.ukf = UnscentedKalmanFilter(dim_x=self.n_states, dim_z=self.n_obs, dt=self.delta_t, 
                                        fx=self.f_k, hx=self.h_k, points=self.sigma_points)
        self.ukf.R = self.R_k.numpy()
        self.ukf.Q = self.Q_k.numpy()
        self.ukf.x = torch.ones((self.n_states,)).numpy()
        self.ukf.P = (torch.eye(self.n_states)*1e-5).numpy()
        return None

    def initialize(self):

        self.ukf.x = torch.ones((self.n_states,)).numpy()
        self.ukf.P = (torch.eye(self.n_states)*1e-5).numpy()

    def push_to_device(self, x):
        """ Push the given tensor to the device
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def get_sigma_points(self):
        self.sigma_points = MerweScaledSigmaPoints(self.n_states, alpha=self.alpha, beta=self.beta, kappa=self.kappa)

    def run_mb_filter(self, X, Y, U=None):

        _, Ty, dy = Y.shape
        _, Tx, dx = X.shape

        if len(Y.shape) == 3:
            N, T, d = Y.shape
        elif len(Y.shape) == 2:
            T, d = Y.shape
            N = 1
            Y = Y.reshape((N, Ty, d))

        traj_estimated_predicted = torch.zeros((N, Tx, self.n_states), device=self.device).type(torch.FloatTensor)
        Pk_estimated_predicted = torch.zeros((N, Tx, self.n_states, self.n_states), device=self.device).type(torch.FloatTensor)
        
        traj_estimated_filtered = torch.zeros((N, Tx, self.n_states), device=self.device).type(torch.FloatTensor)
        Pk_estimated_filtered = torch.zeros((N, Tx, self.n_states, self.n_states), device=self.device).type(torch.FloatTensor)
        
        # MSE [Linear]
        MSE_UKF_linear_arr_filtered = torch.zeros((N,)).type(torch.FloatTensor)
        MSE_UKF_linear_arr_predicted = torch.zeros((N,)).type(torch.FloatTensor)
        # points = JulierSigmaPoints(n=SysModel.m)
        
        start = timer()
        for i in range(0, N):
            self.initialize()
            #if self.init_cond is not None:
            #    self.ukf.x = torch.unsqueeze(self.init_cond[i, :], 1).numpy()
            for k in range(0, Ty):
                
                self.ukf.predict()
                traj_estimated_predicted[i,k+1,:] = torch.from_numpy(self.ukf.x)
                Pk_estimated_predicted[i,k+1,:,:] = torch.from_numpy(self.ukf.P)
                self.ukf.update(Y[i,k,:].numpy())       
                traj_estimated_filtered[i,k+1,:] = torch.from_numpy(self.ukf.x)
                Pk_estimated_filtered[i,k+1,:,:] = torch.from_numpy(self.ukf.P)

            #MSE_UKF_linear_arr[i] = mse_loss(traj_estimated[i], X[i]).item()
            MSE_UKF_linear_arr_predicted[i] = mse_loss(X[i,:,:], traj_estimated_predicted[i,:,:]).mean().item()
            MSE_UKF_linear_arr_filtered[i] = mse_loss(X[i,:,:], traj_estimated_filtered[i,:,:]).mean().item()
            #print("ukf, sample: {}, mse_loss: {}".format(i+1, MSE_UKF_linear_arr[i]))

        end = timer()
        t = end - start

        #MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
        #MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)
        # Standard deviation
        #MSE_UKF_linear_std = torch.std(MSE_UKF_linear_arr, unbiased=True)
        #MSE_UKF_dB_std = 10 * torch.log10(MSE_UKF_linear_std.abs())

        #print("UKF - MSE LOSS:", MSE_UKF_dB_avg, "[dB]")
        #print("UKF - MSE STD:", MSE_UKF_dB_std, "[dB]")

        mse_ukf_dB_avg_predicted = torch.mean(10*torch.log10(MSE_UKF_linear_arr_predicted), dim=0)
        mse_ukf_dB_avg_filtered = torch.mean(10*torch.log10(MSE_UKF_linear_arr_filtered), dim=0)

        print("UKF - MSE LOSS (PREDICTED) :", mse_ukf_dB_avg_predicted, "[dB]")
        print("UKF - MSE STD (PREDICTED):", torch.std(10*torch.log10(MSE_UKF_linear_arr_predicted), dim=0), "[dB]")

        print("UKF - MSE LOSS (FILTERED) :", mse_ukf_dB_avg_filtered, "[dB]")
        print("UKF - MSE STD (FILTERED):", torch.std(10*torch.log10(MSE_UKF_linear_arr_filtered), dim=0), "[dB]")

        # Print Run Time
        #print("Inference Time:", t)

        return traj_estimated_predicted, Pk_estimated_predicted, MSE_UKF_linear_arr_predicted.mean(), mse_ukf_dB_avg_predicted, \
            traj_estimated_filtered, Pk_estimated_filtered, MSE_UKF_linear_arr_filtered.mean(), mse_ukf_dB_avg_filtered