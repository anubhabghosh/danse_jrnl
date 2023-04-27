#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
# This function is used to define the parameters of the model
import numpy as np
import math
import torch
from utils.utils import dB_to_lin, partial_corrupt
import scipy    
from scipy.linalg import block_diag
from ssm_models import LinearSSM
import torch
from torch.autograd.functional import jacobian

torch.manual_seed(10)
delta_t = 0.02 # Hardcoded for now
delta_t_test = 0.04 # Hardcoded for now
J_gen = 5 
J_test = 5 # hardcoded for now

def A_fn(z):
    return np.array([
                    [-10, 10, 0],
                    [28, -1, -z],
                    [0, z, -8.0/3]
                ])

def h_fn(z):
    return z

"""
# The KalmanNet implementation
def f_lorenz(x):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    #A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) #(torch.add(torch.reshape(torch.matmul(B, x),(3,3)).T,C))
    A = (torch.add(torch.reshape(torch.matmul(B, x),(3,3)).T,C))
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)
"""

def f_lorenz_danse_test_ukf(x, dt):

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
        F_add = (torch.matrix_power(A*delta_t_test, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()

def f_lorenz_danse_ukf(x, dt):

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
    
def f_lorenz_danse_test(x):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    #delta_t = 0.02 # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_test, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

def f_lorenz_danse(x):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    #delta_t = 0.02 # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

def f_sinssm_fn(z, alpha=0.9, beta=1.1, phi=0.1*math.pi, delta=0.01):
    return alpha * torch.sin(beta * z + phi) + delta

def h_sinssm_fn(z, a=1, b=1, c=0):
    return a * (b * z + c)

def get_H_DANSE(type_, n_states, n_obs):
    if type_ == "LinearSSM":
        return LinearSSM(n_states=n_states, n_obs=n_obs).H
    elif type_ == "LorenzSSM":
        return np.eye(n_obs, n_states)
    elif type_ == "LorenzSSMn{}".format(n_obs):
        return block_diag(np.eye(n_obs), np.zeros((int(3-n_obs),int(3-n_obs))))
    elif type_ == "SinusoidalSSM":
        return jacobian(h_sinssm_fn, torch.randn(n_states,)).numpy()

def get_parameters(n_states=5, n_obs=5, device='cpu'):

    ssm_parameters_dict = {
        # Parameters of the linear model 
        "LinearSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "gamma":0.8,
            "beta":1.0
        },
        # Parameters of the Lorenz Attractor model
        "LorenzSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t,
            "alpha":0.0, # alpha = 0.0, implies a Lorenz model
            "H":None, # By default, H is initialized to an identity matrix
            "delta_d":0.002,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True
        },
        "LorenzSSMn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t,
            "alpha":0.0, # alpha = 0.0, implies a Lorenz model
            "H":block_diag(np.eye(n_obs), np.zeros((int(3-n_obs),int(3-n_obs)))), # By default, H is initialized to an identity matrix
            "delta_d":0.002,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True
        }
    }

    estimators_dict={
        # Parameters of the DANSE estimator
        "danse":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":None,
            "H":None,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-2,
                    "num_epochs":2000,
                    "min_delta":5e-2,
                    "n_hidden_dense":32,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":50,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                }
            }
        },
        # Parameters of the Model-based filters - KF, EKF, UKF
        "KF":{
            "n_states":n_states,
            "n_obs":n_obs
        },
        "EKF":{
            "n_states":n_states,
            "n_obs":n_obs
        },
        "UKF":{
            "n_states":n_states,
            "n_obs":n_obs,
            "n_sigma":n_states*2,
            "kappa":0.0,
            "alpha":1e-3
        },
        "KNetUoffline":{
            "n_states":n_states,
            "n_obs":n_obs,
            "n_layers":1,
            "N_E":10_0,
            "N_CV":100,
            "N_T":10_0,
            "unsupervised":True,
            "data_file_specification":'Ratio_{}---R_{}---T_{}',
            "model_file_specification":'Ratio_{}---R_{}---T_{}---unsupervised_{}',
            "nu_dB":0.0,
            "lr":1e-3,
            "weight_decay":1e-6,
            "num_epochs":100,
            "batch_size":100,
            "device":device
        }
    }

    return ssm_parameters_dict, estimators_dict
