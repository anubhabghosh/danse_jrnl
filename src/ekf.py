#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
import numpy as np
import torch
import math
from torch import autograd, nn
from utils.utils import dB_to_lin, mse_loss

class EKF(nn.Module):
    """ This class implements an extended Kalman filter in PyTorch
    """
    def __init__(self, n_states, n_obs, J=5, delta=0.02, f=None, h=None, Q=None, R=None, device='cpu', use_Taylor=False):
        super(EKF, self).__init__()

        self.n_states = n_states
        self.device = device

        # Initializing the system model
        self.n_states = n_states # Setting the number of states of the Kalman filter
        self.n_obs = n_obs
        self.J = J # Linear approximation order (Taylor series expansion order J)
        self.delta = delta # Step size for Taylor series expansion of order J
        self.f_k = f # State transition function (relates x_k, u_k to x_{k+1})
        self.h_k = h # Output function (relates state x_k to output y_k)
        self.use_Taylor = use_Taylor # Flag to use Taylor series approximation or not
        
        self.Q_k = self.push_to_device(Q) # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = self.push_to_device(R) # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)

        return None

    def initialize_stats(self):

        # Defining the required state to be estimate 
        self.x_hat_pos_k = torch.ones((self.n_states, 1), device=self.device) # Assuming initial value of the state is zero, i.e. \hat{p}_0^{+} = 0
        self.Pk_pos = torch.eye(self.n_states, device=self.device) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.Sk_pos_sqrt = torch.cholesky(self.Pk_pos).to(self.device) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.x_hat_neg_k = torch.zeros((self.n_states, 1), device=self.device) # Prediction state \hat{x}_{k \vert k-1}
        self.Sk_neg_sqrt = torch.zeros_like(self.Sk_pos_sqrt, device=self.device) # Cholesky factor of the state covariance matrix S_{k \vert k-1} = \sqrt{P_{k \vert k-1}}
        self.Pk_neg = torch.zeros_like(self.Pk_pos, device=self.device) # State covariance matrix P_{k \vert k-1}
        self.Kk = torch.zeros((self.n_states,self.n_obs), device=self.device) # The Kalman gain K_k (filtered version)
        return None

    def push_to_device(self, x):
        """ Push the given tensor to the device
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def f_linearize(self, x):
        
        F_lin = torch.eye(self.n_states)
        for j in range(1, self.J+1):
            F_lin += torch.matrix_power(torch.from_numpy(
                self.f_k(np.asscalar(x[0].numpy()))
                ).type(torch.FloatTensor) * self.delta, j) / math.factorial(j)
        return F_lin

    def predict_estimate(self, Pk_pos_prev, Q_k_prev, u_k=0.0):
        """ This function helps implement the prediction step / time-update step of the Kalman filter, i.e. using 
        available observations, and previous state estimates, what is the next state estimate?

        Args:
            F_k_prev: State transition matrix F_k at time instant k
            Pk_pos_prev: Filtered state covariance matrix P_{k \vert k}
            G_k_prev: Input matrix G_k at time instant k
            Q_k_prev: Process noise covariance Q_k at instant k

        Returns:
            _type_: _description_
        """
        #self.Pk_neg = F_k_prev @ (Pk_pos_prev @ F_k_prev.T) + G_k_prev @ (Q_k_prev @ G_k_prev.T) # Calculating the predicted state covariance P_{k \vert k -1} \equiv P^{-}_{k}
        
        # Implemeting a Square-Root Filtering version for numerical stability
        # Trying QR decomposition to get Pk_neg from Pk_pos
        #F_k_prev = self.compute_jac_f_k(x_=self.x_hat_pos_k[0]).to(self.device).squeeze(-1)
        
        if self.use_Taylor == True: # Case of Lorenz model 
            F_k_prev = self.f_linearize(x=self.x_hat_pos_k).to(self.device)
        else:
            F_k_prev = self.compute_jac_f_k(x_=self.x_hat_pos_k).to(self.device)
            
        self.Sk_pos_sqrt = torch.cholesky(Pk_pos_prev)
        Qk_prev_sqrt = torch.cholesky(Q_k_prev)
        A_state = torch.cat((F_k_prev @ self.Sk_pos_sqrt, Qk_prev_sqrt), dim=1)
        assert A_state.shape[1] == 2*self.n_states, "A_state has a dimension problem"
        _, Ra = torch.qr(A_state.T)
        assert torch.allclose(Ra, torch.triu(Ra)), "Ra is not upper triangular" # check if upper triangular
        self.Sk_neg_sqrt = Ra.T[:,:self.n_states]

        # Time update equation
        self.x_hat_neg_k = self.f_k(self.x_hat_pos_k) # Calculating the predicted state estimate using the previous filtered state estimate \hat{x}_{k-1 \vert k-1}^{+}
        self.Pk_neg = self.Sk_neg_sqrt @ self.Sk_neg_sqrt.T # Calculating the predicted state estimate covariance 

        return self.x_hat_neg_k, self.Pk_neg

    def filtered_estimate(self, y_k):
        """ This function implements the filtering step of the Kalman filter
        Args:
            y_k: The measurement at time instant k 
        Returns:
            _type_: _description_
        """
        
        self.H_k = self.compute_jac_h_k(x_=self.x_hat_neg_k).to(self.device)
        assert self.H_k.shape[1] == self.n_states, "Dimension of H_k is not consistent"

        # Trying to Square root algorithm since K_k is ill conditioned at the moment.
        # So, we use QR decomposition in the measurement equation
        A_measurement = torch.Tensor(np.block([
            [torch.sqrt(self.R_k).numpy() * torch.eye(self.H_k.shape[0]).numpy(), self.H_k.numpy() @ self.Sk_neg_sqrt.numpy()],
            [torch.zeros((self.n_states, self.H_k.shape[0])).numpy(), self.Sk_neg_sqrt.numpy()]
            ])).type(torch.FloatTensor)
        
        Qa_measurement, Ra_measurement = torch.qr(A_measurement.T)
        Re_k_sqrt = Ra_measurement.T[:self.H_k.shape[0],:self.H_k.shape[0]]
        K_k_Re_k_sqrt = Ra_measurement.T[self.H_k.shape[0]:,:self.H_k.shape[0]]
        self.K_k = K_k_Re_k_sqrt @ torch.inverse(Re_k_sqrt)
        assert self.K_k.shape[0] == self.n_states, "Kalman gain has a dimension issue"

        self.Sk_pos_sqrt = Ra_measurement.T[self.H_k.shape[0]:,self.H_k.shape[0]:]
        
        # Measurement update equation
        self.x_hat_pos_k = self.x_hat_neg_k + self.K_k @ (y_k - self.h_k(self.x_hat_neg_k))
        self.Pk_pos = self.Sk_pos_sqrt @ self.Sk_pos_sqrt.T
        assert self.Pk_pos.shape[0] == self.n_states and self.Pk_pos.shape[1] == self.n_states, "Dimension problem for P_k"

        return self.x_hat_pos_k, self.Pk_pos

    def compute_jac_f_k(self, x_, inputs_=0.0):
        x_ = x_.reshape((-1,))
        F_k = autograd.functional.jacobian(self.f_k, x_)
        return F_k

    def compute_jac_h_k(self, x_):
        x_ = x_.reshape((-1,))
        H_k = autograd.functional.jacobian(self.h_k, x_)
        return H_k

    def run_mb_filter(self, X, Y):
        
        _, Ty, dy = Y.shape
        _, Tx, dx = X.shape

        if len(Y.shape) == 3:
            N, Ty, d = Y.shape
        elif len(Y.shape) == 2:
            Ty, d = Y.shape
            N = 1
            Y = Y.reshape((N, Ty, d))

        traj_estimated = torch.zeros((N, Tx, self.n_states), device=self.device).type(torch.FloatTensor)
        Pk_estimated = torch.zeros((N, Tx, self.n_states, self.n_states), device=self.device).type(torch.FloatTensor)
        mse_arr = torch.zeros((N,)).type(torch.FloatTensor)

        for i in range(0, N):
            self.initialize_stats()
            for k in range(0, Ty):

                x_rec_hat_neg_k, Pk_neg = self.predict_estimate(Pk_pos_prev=self.Pk_pos, Q_k_prev=self.Q_k)
                
                x_rec_hat_pos_k, Pk_pos = self.filtered_estimate(y_k=Y[i,k].view(-1,1))
            
                # Save filtered state estimates
                traj_estimated[i,k+1,:] = x_rec_hat_pos_k.view(-1,)
                
                #Also save covariances
                Pk_estimated[i,k+1,:,:] = Pk_pos

            mse_arr[i] = mse_loss(X[i,1:,:], traj_estimated[i,1:,:]).mean()  # Calculate the squared error across the length of a single sequence
            #print("ekf, sample: {}, mse_loss: {}".format(i+1, mse_arr[i]))

        #mse_ekf_lin_avg = torch.mean(10*torch.log10(mse_arr), dim=0) # Calculate the MSE by averaging over all examples in a batch
        mse_ekf_dB_avg = torch.mean(10*torch.log10(mse_arr), dim=0)
        print("EKF - MSE LOSS:", mse_ekf_dB_avg, "[dB]")
        print("EKF - MSE STD:", torch.std(10*torch.log10(mse_arr), dim=0), "[dB]")

        return traj_estimated, Pk_estimated, mse_ekf_dB_avg