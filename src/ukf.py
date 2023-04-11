######################################################################## 
# Implementing an Unscented Kalman filter - SQ
#########################################################################
import numpy as np
import torch
from torch import autograd, nn
from utils.utils import dB_to_lin, mse_loss

class UKF(nn.Module):
    """ This class implements an unscented Kalman filter in PyTorch
    """
    def __init__(self, n_states, n_obs, f=None, h=None, Q=None, R=None, kappa=0, alpha=1e-3, beta=2, n_sigma=None, inverse_r2_dB=None, nu_dB=None, device='cpu'):
        super(UKF, self).__init__()

        # Initialize the device
        self.device = device

        # Initializing the system model
        self.n_states = n_states # Setting the number of states of the Kalman filter
        self.n_obs = n_obs
        self.f_k = f # State transition function (relates x_k, u_k to x_{k+1})
        self.h_k = h # Output function (relates state x_k to output y_k)
         
        if (not inverse_r2_dB is None) and (not nu_dB is None):
            r2 = 1.0 / dB_to_lin(inverse_r2_dB)
            q2 = dB_to_lin(nu_dB - inverse_r2_dB)
            Q = q2 * np.eye(self.n_states)
            R = r2 * np.eye(self.n_obs)

        self.Q_k = self.push_to_device(Q) # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = self.push_to_device(R) # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)
        
        # Defining the required state to be estimate 
        self.x_hat_pos_k = torch.zeros((self.n_states, 1), device=self.device) # Assuming initial value of the state is zero, i.e. \hat{p}_0^{+} = 0
        self.Pk_pos = torch.eye(self.n_states, device=self.device) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.Sk_pos_sqrt = torch.cholesky(self.Pk_pos).to(self.device) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.x_hat_neg_k = torch.empty((self.n_states, 1), device=self.device) # Prediction state \hat{x}_{k \vert k-1}
        self.Sk_neg_sqrt = torch.empty_like(self.Sk_pos_sqrt, device=self.device) # Cholesky factor of the state covariance matrix S_{k \vert k-1} = \sqrt{P_{k \vert k-1}}
        self.Pk_neg = torch.empty_like(self.Pk_pos, device=self.device) # State covariance matrix P_{k \vert k-1}
        self.Kk = torch.zeros((self.n_states,self.n_obs), device=self.device) # The Kalman gain K_k (filtered version)

        # Defining the parameters for the unscented transformation
        self.kappa = kappa # Hyperparameter in the UT
        self.alpha = alpha # Hyperparameter in the UT
        self.beta = beta # Hyperparameter in the UT

        if not n_sigma is None:
            self.n_sigma = n_sigma
        else:
            self.n_sigma = 2 * self.n_states + 1

        self.x_hat_neg_k_sigma = torch.empty((self.n_states, self.n_sigma), device=self.device)
        self.x_hat_pos_k_sigma = torch.empty((self.n_states, self.n_sigma), device=self.device)

        self.lambda_ = self.alpha ** 2 * ((self.n_sigma // 2) + self.kappa) - (self.n_sigma // 2) # Hyperparameter in the UT

        self.set_mean_weights()
        self.set_cov_weights()

        return None

    def push_to_device(self, x):
        """ Push the given tensor to the device
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def set_mean_weights(self):
        self.W_m = torch.zeros((self.n_sigma,), device=self.device)
        self.W_m[0] = self.lambda_ / ((self.n_sigma // 2) + self.lambda_)
        for i in range(1, self.n_sigma):
            self.W_m[i] = 1.0 / (2 * ((self.n_sigma // 2) + self.lambda_))
    
    def set_cov_weights(self):
        self.W_c = torch.zeros((self.n_sigma,), device=self.device)
        self.W_c[0] = (self.lambda_ / ((self.n_sigma // 2) + self.lambda_)) + (1 - self.alpha**2 + self.beta)
        for i in range(1, self.n_sigma):
            self.W_c[i] = 1.0 / (2 * ((self.n_sigma // 2) + self.lambda_))

    def unscented_transform(self, xk, Pk):
        
        Sigma_tilde_points = torch.zeros_like(self.x_hat_neg_k_sigma, device=self.device)
        for i in range(1, self.n_sigma):
            if i <= self.n_sigma // 2:
                Sigma_tilde_points[:, i] = torch.transpose(torch.cholesky((self.n_sigma // 2 + self.lambda_)*Pk),0,1)[i-1, :]
            else:
                Sigma_tilde_points[:, i] = torch.transpose(-torch.cholesky((self.n_sigma // 2 + self.lambda_)*Pk),0,1)[i-(self.n_sigma // 2)-1,:]
        
        return Sigma_tilde_points + torch.repeat_interleave(xk.reshape((-1,1)), self.n_sigma, dim=1)

    def transform_sigma_points_prediction(self, u_k=0):
        transformed_sigma_points_prediction = torch.zeros_like(self.x_hat_pos_k_sigma, device=self.device)
        for i in range(self.x_hat_pos_k_sigma.shape[1]):
            transformed_sigma_points_prediction[:, i] = (torch.from_numpy(self.f_k(self.x_hat_pos_k_sigma[0,i])).type(torch.FloatTensor).to(self.device) @ self.x_hat_pos_k_sigma[:,i].view(-1,1)).view(-1,)
        return transformed_sigma_points_prediction

    def transform_sigma_points_filtering(self):
        transformed_sigma_points_filtering = torch.zeros_like(self.x_hat_neg_k_sigma, device=self.device)
        for i in range(self.x_hat_neg_k_sigma.shape[1]):
            transformed_sigma_points_filtering[:, i] = self.h_k(self.x_hat_neg_k_sigma[:,i])
        return transformed_sigma_points_filtering
        #return self.h_k(self.x_hat_neg_k_sigma)    

    def predict_estimate(self, Q_k_prev, u_k=0.0):
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
        self.x_hat_pos_k_sigma = self.unscented_transform(self.x_hat_pos_k, self.Pk_pos)
        self.x_hat_pos_k_sigma_transformed = self.transform_sigma_points_prediction(u_k)
        self.x_hat_neg_k = torch.mean(
            self.x_hat_pos_k_sigma_transformed * self.W_m.reshape((1,-1)),
            dim=1
        ).view((self.n_states,1))
        #self.Pk_neg = (1.0/self.n_sigma) * (self.x_hat_pos_k_sigma_transformed - self.x_hat_neg_k) @ (self.x_hat_pos_k_sigma_transformed - self.x_hat_neg_k).T
        x_dev = (self.x_hat_pos_k_sigma_transformed - self.x_hat_neg_k)
        self.Pk_neg = x_dev @ (x_dev * self.W_c.reshape((1,-1))).T
        self.Pk_neg += Q_k_prev
        return self.x_hat_neg_k, self.Pk_neg

    def filtered_estimate(self, y_k):
        """ This function implements the filtering step of the Kalman filter
        Args:
            y_k: The measurement at time instant k 
        Returns:
            _type_: _description_
        """
        
        self.x_hat_neg_k_sigma = self.unscented_transform(self.x_hat_neg_k, self.Pk_neg)
        self.x_hat_neg_k_sigma_transformed = self.transform_sigma_points_filtering()
        y_hat_k = torch.mean(
            self.x_hat_neg_k_sigma_transformed * self.W_m.reshape((1,-1)),
            dim=1
        ).reshape((-1,1))
        x_dev = (self.x_hat_neg_k_sigma - self.x_hat_neg_k)
        y_dev = (self.x_hat_neg_k_sigma_transformed - y_hat_k)
        #Py = (1.0/self.n_sigma) * (self.x_hat_neg_k_sigma_transformed - y_hat_k) @ (self.x_hat_neg_k_sigma_transformed - y_hat_k).T
        Py = y_dev @ (y_dev * self.W_c.reshape((1,-1))).T
        Py += self.R_k
        #Pxy = (1.0/self.n_sigma) * (self.x_hat_neg_k_sigma - self.x_hat_neg_k) @ (self.x_hat_neg_k_sigma_transformed - y_hat_k).T
        Pxy = x_dev @ (y_dev * self.W_c.reshape((1,-1))).T
        self.K_k = Pxy @ torch.inverse(Py)
        self.x_hat_pos_k = self.x_hat_neg_k + self.K_k @ (y_k - y_hat_k)
        self.Pk_pos = self.Pk_neg - self.K_k @ Py @ self.K_k.T
        return self.x_hat_pos_k, self.Pk_pos
    
    def run_mb_filter(self, X, Y, U=None):

        _, Ty, dy = Y.shape
        _, Tx, dx = X.shape

        if len(Y.shape) == 3:
            N, T, d = Y.shape
        elif len(Y.shape) == 2:
            T, d = Y.shape
            N = 1
            Y = Y.reshape((N, Ty, d))

        traj_estimated = torch.zeros((N, Tx, self.n_states), device=self.device).type(torch.FloatTensor)
        Pk_estimated = torch.zeros((N, Tx, self.n_states, self.n_states), device=self.device).type(torch.FloatTensor)
        mse_arr = torch.zeros((N,)).type(torch.FloatTensor)

        for i in range(0, N):
            for k in range(0, T):

                x_rec_hat_neg_k, Pk_neg = self.predict_estimate(Q_k_prev=self.Q_k)
                x_rec_hat_pos_k, Pk_pos = self.filtered_estimate(y_k=Y[i,k].view(-1,1))
            
                # Save filtered state estimates
                traj_estimated[i,k+1,:] = x_rec_hat_pos_k.view(-1,)
                
                # Also save covariances
                Pk_estimated[i,k+1,:,:] = Pk_pos
                
            mse_arr[i] = mse_loss(traj_estimated[i], X[i])  # Calculate the squared error across the length of a single sequence
            print("batch: {}, mse_loss: {}".format(i+1, mse_arr[i]))

        mse = torch.mean(mse_arr, dim=0) # Calculate the MSE by averaging over all examples in a batch
        return traj_estimated, Pk_estimated, mse