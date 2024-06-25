#########################################################################
# This script creates a class and associated fucntions for implementation
# of a class of structured inference networks - Deep Markov models (DMM) 
# as per the paper: 

# [1]. Krishnan, Rahul, Uri Shalit, and David Sontag. 
# "Structured inference networks for nonlinear state space models." 
# Proceedings of the AAAI Conference on Artificial Intelligence. 
# Vol. 31. No. 1. 2017. 

# Parts of this implementation has been adapted from
# https://github.com/yjlolo/pytorch-deep-markov-model/blob/master/

# NOTE: The concerned implementations here relate to the use of DMMs in causal 
# mode i.e. we consider only inference implmentations that involve predicting 
# a distribution of x_t given all information in the past such as 
# y_{1:t} (and x_{t-1}). As per [1], this means using a DMM in either MF-L or 
# ST-L mode.

# Creator: Anubhab Ghosh, Jan 2024
#########################################################################

# Import the necessary libraries
import sys
import copy
import os
from os import path
# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim, distributions
from timeit import default_timer as timer
from utils.utils import *
from timeit import default_timer as timer
from utils.losses import dmm_loss_seq
from utils.plot_functions import plot_losses

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    return None

def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets

class DMM(nn.Module):

    def __init__(self, 
                obs_dim=3,  # Dimension of the observation / input to RNN
                latent_dim=3, # Dimension of the latent state / output of RNN in case of state estimation
                rnn_model_type='gru', # Type of RNN used (GRU / LSTM / RNN)
                rnn_params_dict=None, # The dictionary to initialize RNN parameters
                batch_size=64, # Training batch size
                optimizer_params=None, # Dictionary to initialize / set the optimizer parameters
                use_mean_field_q=False, # Flag to indicate the use of mean-field q(x_{1:T} \vert y_{1:T})
                inference_mode='st-l', # String to indicate the type of DMM inference mode (typically, we will use ST-L or MF-L)
                combiner_dim=40, # Dimension of hidden layer of combiner network
                train_emission=False, # Flag to indicate if emission network needs to be learned (True) or not (False)
                H=None, # Measurement matrix, in case of nontrainable emission network with linear measurement
                C_w=None, # Measurmenet noise cov. matrix, in case of nontrainable emission network with linear measurements
                emission_dim=40, # Dimension of hidden layer for emission network
                emission_num_layers=2, # No. of hidden layers for emission network
                emission_use_binary_obs=False, # Flag to indicate the modeling of binary observations or not
                train_transition=True, # Flag to indicate if transition network needs to be learned (True) or not (False)
                transition_dim=40, # Dimension of hidden layer for transition network
                transition_num_layers=2, # No. of hidden layers for transition network
                train_initials=False, # Set if the initial states also are learned uring the optimization 
                device='cpu'
                ):
        super(DMM, self).__init__()
        
        # Initialize DMM variables
        self.obs_dim = obs_dim # This is the input dimension (in case of inference, this is the observation dimension)
        self.latent_dim = latent_dim # This is the dimension of the hidden state of SSM
        self.train_initials = train_initials
        self.device = device

        # Initializing the combiner network
        self.use_mean_field_q = use_mean_field_q 
        self.inference_mode = inference_mode
        self.combiner_dim = combiner_dim
        self.init_combiner_net()
        self.latent_q_0 = self.init_latent_seq_combiner_0(trainable=self.train_initials)

        # Initializing the optimizer parameters
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size

        # initializing the inference network parameters 
        self.rnn_type = rnn_model_type
        self.rnn_h_dim = rnn_params_dict[self.rnn_type]['n_hidden']
        self.rnn_num_layers = rnn_params_dict[self.rnn_type]['n_layers']
        self.rnn_use_bidirectional = rnn_params_dict[self.rnn_type]['bidirectional']
        self.rnn_num_directions = 1 if not self.rnn_use_bidirectional else 2
        self.rnn_use_batch_first = rnn_params_dict[self.rnn_type]['batch_first']
        self.rnn_dropout = rnn_params_dict[self.rnn_type]['dropout']
        self.rnn_use_bias = rnn_params_dict[self.rnn_type]['bias']
        self.init_rnn()

        # initializing the emission network parameters 
        self.train_emission = train_emission
        self.H = H
        self.C_w = C_w
        self.emission_dim = emission_dim
        self.emission_num_layers = emission_num_layers
        self.emission_use_binary_obs = emission_use_binary_obs
        self.init_emission_net(self.train_emission)

        # initializing the transition network parameters
        self.train_transition = train_transition
        self.transition_dim = transition_dim
        self.tranisition_num_layers = transition_num_layers
        self.init_transition_net()
        self.mu_latent_0, self.var_latent_0 = self.init_latent_seq_0(trainable=self.train_initials)
    
    def push_to_device(self, x):
        """ Push the given tensor to the device
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
    
    def init_rnn(self):

        # Defining the recurrent layers 
        if self.rnn_type.lower() == "rnn": # RNN 
            self.rnn = nn.RNN(input_size=self.obs_dim, hidden_size=self.rnn_h_dim, 
                num_layers=self.rnn_num_layers, batch_first=self.rnn_use_batch_first,
                bidirectional=self.rnn_use_bidirectional, dropout=self.rnn_dropout, bias=self.rnn_use_bias)   
        elif self.rnn_type.lower() == "lstm": # LSTM
            self.rnn = nn.LSTM(input_size=self.obs_dim, hidden_size=self.rnn_h_dim, 
                num_layers=self.rnn_num_layers, batch_first=self.rnn_use_batch_first,
                bidirectional=self.rnn_use_bidirectional, dropout=self.rnn_dropout, bias=self.rnn_use_bias)
        elif self.rnn_type.lower() == "gru": # GRU
            self.rnn = nn.GRU(input_size=self.obs_dim, hidden_size=self.rnn_h_dim, 
                num_layers=self.rnn_num_layers, batch_first=self.rnn_use_batch_first,
                bidirectional=self.rnn_use_bidirectional, dropout=self.rnn_dropout, bias=self.rnn_use_bias)  
        else:
            print("Model type unknown:", self.rnn_type.lower()) 
            sys.exit()
        
    def init_rnn_hidden(self, trainable=True):
        if self.rnn_type.lower() == 'lstm':
            h0 = nn.Parameter(torch.zeros(self.rnn_num_layers * self.rnn_num_directions, 1, self.rnn_h_dim), requires_grad=trainable)
            c0 = nn.Parameter(torch.zeros(self.rnn_num_layers * self.rnn_num_directions, 1, self.rnn_h_dim), requires_grad=trainable)
            return h0, c0
        else:
            h0 = nn.Parameter(torch.zeros(self.rnn_num_layers * self.rnn_num_directions, 1, self.rnn_h_dim), requires_grad=trainable)
            return h0
        
    def encode_rnn(self, obs_seq):
        
        h_rnn_seq, _ = self.rnn(obs_seq)
        return h_rnn_seq

    def init_emission_net(self, trainable=False):

        if trainable == True:
            self.emission_net = nn.Sequential()
            self.emission_net.add_module(name='l{}'.format(1), module=nn.Linear(self.latent_dim, self.emission_dim))
            self.emission_net.add_module(name='nl{}'.format(1), module=nn.ReLU())

            for layer_i in range(1, self.emission_num_layers-1):
                self.emission_net.add_module(name='l{}'.format(layer_i+1), module=nn.Linear(self.emission_dim, self.emission_dim))
                self.emission_net.add_module(name='nl{}'.format(layer_i+1), module=nn.ReLU())

            if self.emission_use_binary_obs == True:
                self.emission_net.add_module(name='l{}'.format(self.emission_num_layers), module=nn.Linear(self.emission_dim, self.obs_dim))
            else:
                self.emission_net.add_module(name='l{}'.format(self.emission_num_layers), module=nn.Linear(self.emission_dim, int(2*self.obs_dim)))
        else:
            self.H = self.push_to_device(self.H).requires_grad_(False)
            self.C_w = self.push_to_device(self.C_w).requires_grad_(False)

    def init_latent_seq_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.latent_dim), requires_grad=trainable), \
            nn.Parameter(torch.ones(self.latent_dim), requires_grad=trainable)

    def forward_emission_net(self, latent_seq_current):

        if self.emission_use_binary_obs == True:
            prob_obs_given_latent_seq_curr = torch.sigmoid(self.emission_net.forward(latent_seq_current))
            return prob_obs_given_latent_seq_curr

        else:
            if self.train_emission == True:
                mu_var_obs_given_latent_seq_curr = self.emission_net.forward(latent_seq_current)
                mu_obs_given_latent_seq_curr = mu_var_obs_given_latent_seq_curr[:, :, :self.obs_dim]
                var_obs_given_latent_seq_curr = F.softplus(mu_var_obs_given_latent_seq_curr[:, :, self.obs_dim:])
            else:
                mu_obs_given_latent_seq_curr = torch.einsum('ij,ntj->nti',self.H, latent_seq_current)
                var_obs_given_latent_seq_curr = torch.repeat_interleave(
                    torch.repeat_interleave(
                        torch.diag(self.C_w).reshape((1,1,-1)), latent_seq_current.shape[1], dim=1
                    ), 
                    latent_seq_current.shape[0], dim=0
                    )
            return mu_obs_given_latent_seq_curr, var_obs_given_latent_seq_curr

    def init_transition_net(self):
        
        self.Wg1 = nn.Linear(self.latent_dim, self.transition_dim)
        self.Wg2 = nn.Linear(self.transition_dim, self.latent_dim)
        self.Wh1 = nn.Linear(self.latent_dim, self.transition_dim)
        self.Wh2 = nn.Linear(self.transition_dim, self.latent_dim)
        self.W_var_latent = nn.Linear(self.latent_dim, self.latent_dim)

    def forward_transition_net(self, latent_seq_prev):
        
        g_seq_curr = torch.sigmoid(self.Wg2(F.relu(self.Wg1(latent_seq_prev))))
        h_seq_curr = self.Wh2(F.relu(self.Wh1(latent_seq_prev)))
        mu_latent_given_latent_prev = g_seq_curr * h_seq_curr + (1 - g_seq_curr) * (latent_seq_prev)
        var_latent_given_latent_prev = F.softplus(self.W_var_latent(F.relu(h_seq_curr)))
        return mu_latent_given_latent_prev, var_latent_given_latent_prev
    
    def init_latent_seq_combiner_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.latent_dim), requires_grad=trainable)
    
    def init_combiner_net(self):

        # initializing the combiner network parameters
        if self.use_mean_field_q == False:
            self.W_h_combined_given_prev_plus_obs = nn.Linear(self.latent_dim, self.combiner_dim)
            self.W_mu_latent_q_given_h_combined = nn.Linear(self.combiner_dim, self.latent_dim)
            self.W_var_latent_q_given_h_combined = nn.Linear(self.combiner_dim, self.latent_dim)
        else:
            self.W_mu_latent_q_given_h_combined = nn.Linear(self.combiner_dim, self.latent_dim)
            self.W_var_latent_q_given_h_combined = nn.Linear(self.combiner_dim, self.latent_dim)
            if self.inference_mode == 'mf-lr':
                self.W_mu_latent_q_given_h_combined_r = nn.Linear(self.combiner_dim, self.latent_dim)
                self.W_var_latent_q_given_h_combined_r = nn.Linear(self.combiner_dim, self.latent_dim)

    def combiner_function(self, h_rnn_seq_curr_instant, latent_sampled_prev_instant=None):

        #h_rnn_seq = self.encode_rnn(obs_seq)
        if self.use_mean_field_q == False:
            assert not latent_sampled_prev_instant is None == True, "Ancestral sampling for approx. posterior needs to be carried out first !!"
            # This means that we are using structured inference modes
            if self.inference_mode == 'st-l':
                assert h_rnn_seq_curr_instant.shape[-1] == self.rnn_num_directions * self.rnn_h_dim
                h_rnn_curr_instant_left = h_rnn_seq_curr_instant[:, :self.rnn_h_dim]
                h_seq_combined = (1.0 / 2) * torch.tanh((self.W_h_combined_given_prev_plus_obs(latent_sampled_prev_instant)) + h_rnn_curr_instant_left)
            elif self.inference_mode == 'st-r':
                assert h_rnn_seq_curr_instant.shape[-1] == self.rnn_num_directions * self.rnn_h_dim
                h_rnn_curr_instant_right = h_rnn_seq_curr_instant[:, self.rnn_h_dim:]
                h_seq_combined = (1.0 / 2) * torch.tanh((self.W_h_combined_given_prev_plus_obs(latent_sampled_prev_instant)) + h_rnn_curr_instant_right)
            elif self.inference_mode == 'st-lr' or self.inference_mode == 'dks':
                assert h_rnn_seq_curr_instant.shape[-1] == self.rnn_num_directions * self.rnn_h_dim
                h_rnn_curr_instant_right = h_rnn_seq_curr_instant[:, self.rnn_h_dim:]
                h_rnn_curr_instant_left = h_rnn_seq_curr_instant[:, :self.rnn_h_dim]
                h_seq_combined = (1.0 / 3) * torch.tanh((self.W_h_combined_given_prev_plus_obs(latent_sampled_prev_instant)) + h_rnn_curr_instant_right + h_rnn_curr_instant_left)

            mu_latent_q_given_h_combined = self.W_mu_latent_q_given_h_combined(h_seq_combined)
            var_latent_q_given_h_combined = F.softplus(self.W_mu_latent_q_given_h_combined(h_seq_combined))
        
        else:
            # This means that we are using mean-field inference modes
            if self.inference_mode == 'mf-l':
                assert h_rnn_seq_curr_instant.shape[-1] == self.rnn_num_directions * self.rnn_h_dim
                h_seq_combined = h_rnn_seq_curr_instant[:, :self.rnn_h_dim] 
                mu_latent_q_given_h_combined = self.W_mu_latent_q_given_h_combined(h_seq_combined)
                var_latent_q_given_h_combined = F.softplus(self.W_mu_latent_q_given_h_combined(h_seq_combined))
            elif self.inference_mode == 'mf-r':
                assert h_rnn_seq_curr_instant.shape[-1] == self.rnn_num_directions * self.rnn_h_dim
                h_seq_combined = h_rnn_seq_curr_instant[:, self.rnn_h_dim:]
                mu_latent_q_given_h_combined = self.W_mu_latent_q_given_h_combined(h_seq_combined)
                var_latent_q_given_h_combined = F.softplus(self.W_mu_latent_q_given_h_combined(h_seq_combined))
            elif self.inference_mode == 'mf-lr':
                assert h_rnn_seq_curr_instant.shape[-1] == self.rnn_num_directions * self.rnn_h_dim
                h_rnn_seq_curr_instant_right = h_rnn_seq_curr_instant[:, self.rnn_h_dim:]
                h_rnn_seq_curr_instant_left = h_rnn_seq_curr_instant[:, :self.rnn_h_dim]
                mu_latent_q_given_h_combined_left = self.W_mu_latent_q_given_h_combined(h_rnn_seq_curr_instant_left)
                var_latent_q_given_h_combined_left = F.softplus(self.W_mu_latent_q_given_h_combined(h_rnn_seq_curr_instant_left))
                mu_latent_q_given_h_combined_right = self.W_mu_latent_q_given_h_combined_r(h_rnn_seq_curr_instant_right)
                var_latent_q_given_h_combined_right = F.softplus(self.W_mu_latent_q_given_h_combined_r(h_rnn_seq_curr_instant_right))

                mu_latent_q_given_h_combined = torch.divide((mu_latent_q_given_h_combined_left * var_latent_q_given_h_combined_left + mu_latent_q_given_h_combined_right * var_latent_q_given_h_combined_right),
                                                            (var_latent_q_given_h_combined_left + var_latent_q_given_h_combined_right))
                var_latent_q_given_h_combined = torch.divide((var_latent_q_given_h_combined_left * var_latent_q_given_h_combined_right),
                                                            (var_latent_q_given_h_combined_left + var_latent_q_given_h_combined_right))

        return mu_latent_q_given_h_combined, var_latent_q_given_h_combined
    
    def reparameterization(self, mu_seq, var_seq):
        # Implementing the reparametrization trick for Gaussians 
        return mu_seq + torch.sqrt(var_seq) * torch.randn_like(var_seq)
    
    def inference(self, input_batch):
        
        batch_size, seq_len, _ = input_batch.shape
        h_rnn_seq = self.encode_rnn(input_batch)
        latent_q_0 = self.latent_q_0.expand(batch_size, self.latent_dim)
        mu_latent_0 = self.mu_latent_0.expand(batch_size, 1, self.latent_dim)
        var_latent_0 = self.var_latent_0.expand(batch_size, 1, self.latent_dim)
        latent_prev_instant = latent_q_0

        #obs_recon_seq = torch.zeros([batch_size, seq_len, self.obs_dim]).to(self.device)
        mu_latent_q_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        var_latent_q_seq = torch.ones([batch_size, seq_len, self.latent_dim]).to(self.device)
        mu_latent_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        var_latent_seq = torch.ones([batch_size, seq_len, self.latent_dim]).to(self.device)
        latent_q_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        #latent_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)

        for t in range(seq_len):
            
            # q(z_t | z_{t-1}, x_{1:t}) or q(z_t | z_{t-1}, x_{t:T}) or q(z_t | z_{t-1}, x_{1:T}) or
            # q(z_t | x_{1:t}) or q(z_t | x_{t:T}) or q(z_t | x_{1:T})
            mu_latent_q_instant, var_latent_q_instant = self.combiner_function(
                h_rnn_seq_curr_instant=h_rnn_seq[:, t, :], 
                latent_sampled_prev_instant=latent_prev_instant)
            
            # NOTE: Ideally, we should be utilizing the reparameterization trick and sampling, but in practice
            # the expectations (in cascade) are approximated by a single realization, for a smaller variance,
            # we are using directly the mean.
            #latent_q_instant = mu_latent_q_instant #self.reparameterization(mu_latent_q_instant, var_latent_q_instant)
            latent_q_instant = self.reparameterization(mu_latent_q_instant, var_latent_q_instant)
            latent_prev_instant = latent_q_instant.clone()
            # p(z_t | z_{t-1})
            mu_latent_curr_instant, var_latent_curr_instant = self.forward_transition_net(latent_prev_instant)
            #latent_curr_instant = self.reparameterization(mu_latent_curr_instant, var_latent_curr_instant)
            #obs_recon_curr_instant = self.forward_emission_net(latent_curr_instant).contiguous()

            mu_latent_q_seq[:, t, :] = mu_latent_q_instant
            var_latent_q_seq[:, t, :] = var_latent_q_instant
            latent_q_seq[:, t, :] = latent_q_instant
            mu_latent_seq[:, t, :] = mu_latent_curr_instant
            var_latent_seq[:, t, :] = var_latent_curr_instant
            #latent_seq[:, t, :] = latent_curr_instant
            #obs_recon_seq[:, t, :] = obs_recon_curr_instant

        mu_latent_seq = torch.cat([mu_latent_0, mu_latent_seq[:, :-1, :]], dim=1)
        var_latent_seq = torch.cat([var_latent_0, var_latent_seq[:, :-1, :]], dim=1)
        #latent_0 = self.reparameterization(mu_latent_0, var_latent_0)
        #latent_seq = torch.cat([latent_0, latent_seq[:, :-1, :]], dim=1)

        return latent_q_seq, mu_latent_q_seq, var_latent_q_seq, mu_latent_seq, var_latent_seq

    def generate(self, batch_size, seq_len):

        mu_latent = self.mu_latent_0.expand(batch_size, self.latent_dim)
        var_latent = self.var_latent_0.expand(batch_size, self.latent_dim)
        latent_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        mu_latent_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        var_latent_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        output_obs_seq = torch.zeros([batch_size, seq_len, self.latent_dim]).to(self.device)
        
        for t in range(seq_len):
            
            mu_latent_seq[:, t, :] = mu_latent
            var_latent_seq[:, t, :] = var_latent
            latent_curr_instant = self.reparameterization(mu_latent, var_latent)
            obs_recon_curr_instant = self.forward_emission_net(latent_curr_instant)
            mu_latent, var_latent = self.forward_transition_net(latent_curr_instant)
            output_obs_seq[:, t, :] = obs_recon_curr_instant
            latent_seq[:, t, :] = latent_curr_instant

        return output_obs_seq, latent_seq, mu_latent_seq, var_latent_seq

    def forward(self, obs_seq):

        latent_q_seq, mu_latent_q_seq, var_latent_q_seq, mu_latent_seq, var_latent_seq = self.inference(input_batch=obs_seq)
        return latent_q_seq, mu_latent_q_seq, var_latent_q_seq, mu_latent_seq, var_latent_seq

def train_dmm(model, options, optimizer_params, train_loader, val_loader, logfile_path, 
            modelfile_path, save_chkpoints, device='cpu', tr_verbose=False):

    # Push the model to device and count parameters
    optimizer_params = model.optimizer_params
    nepochs = optimizer_params['num_epochs']
    model = push_model(nets=model, device=device)
    total_num_params, total_num_trainable_params = count_params(model)
    
    # Set the model to training
    model.train()
    mse_criterion = nn.MSELoss()
    if optimizer_params['type'] == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), **optimizer_params['args'])
    else:
        print("Adam should be the recommended optimizer for training!")
        sys.exit(1)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//3, gamma=0.9) # gamma was initially 0.9
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_params['num_epochs']//6, gamma=0.9) # gamma is now set to 0.8
    tr_kl_losses = []
    tr_nll_losses = []
    tr_nvlb_losses = []
    val_kl_losses = []
    val_nll_losses = []
    val_nvlb_losses = []

    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    #if save_chkpoints == True:
    if save_chkpoints == "all" or save_chkpoints == "some":
        # No grid search
        if logfile_path is None:
            training_logfile = "./log/dmm_{}_{}.log".format(model.inference_mode, model.rnn_type)
        else:
            training_logfile = logfile_path

    elif save_chkpoints == None:
        # Grid search
        if logfile_path is None:
            training_logfile = "./log/gs_training_dmm_{}_{}.log".format(model.inference_mode, model.rnn_type)
        else:
            training_logfile = logfile_path
    
    # Call back parameters
    
    patience = 0
    num_patience = 3 
    min_delta = optimizer_params["min_delta"] # 1e-3 for simpler model, for complicated model we use 1e-2
    #min_tol = 1e-3 # for tougher model, we use 1e-2, easier models we use 1e-5
    check_patience=False
    best_val_loss = np.inf
    tr_loss_for_best_val_loss = np.inf
    best_model_wts = None
    best_val_epoch = None
    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp
    

    # Convergence monitoring (checks the convergence but not ES of the val_loss)
    model_monitor = ConvergenceMonitor(tol=min_delta,
                                    max_epochs=num_patience)

    # This checkes the ES of the val loss, if the loss deteriorates for specified no. of
    # max_epochs, stop the training
    #model_monitor = ConvergenceMonitor_ES(tol=min_tol, max_epochs=num_patience)

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))

    # Start time
    starttime = timer()
    try:
        for epoch in range(nepochs):
            
            tr_running_loss = 0.0
            tr_kl_loss_epoch_sum = 0.0
            val_kl_loss_epoch_sum = 0.0
            tr_nll_loss_epoch_sum = 0.0
            val_nll_loss_epoch_sum = 0.0
            tr_nvlb_loss_epoch_sum = 0.0
            val_nvlb_loss_epoch_sum = 0.0
            val_mse_loss_epoch_sum = 0.0
        
            for i, data in enumerate(train_loader, 0):
            
                tr_Y_batch, tr_X_batch = data
                optimizer.zero_grad()
                Y_train_batch = Variable(tr_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                tr_latent_q_seq, tr_mu_latent_q_seq, tr_var_latent_q_seq, tr_mu_latent_seq, tr_var_latent_seq = model.forward(Y_train_batch)
                tr_mu_obs_given_latent_seq_curr, tr_var_obs_given_latent_seq_curr = model.forward_emission_net(
                    latent_seq_current=tr_latent_q_seq
                )
                tr_kl_loss_seq_mean_batch, tr_ll_loss_seq_mean_batch, tr_nvlb_seq_mean_batch = dmm_loss_seq(
                    obs_seq=Y_train_batch,
                    mu_obs_seq=tr_mu_obs_given_latent_seq_curr,
                    var_obs_seq=tr_var_obs_given_latent_seq_curr,
                    mu_latent_q_seq=tr_mu_latent_q_seq,
                    var_latent_q_seq=tr_var_latent_q_seq,
                    mu_latent_seq=tr_mu_latent_seq,
                    var_latent_seq=tr_var_latent_seq
                )
                tr_nvlb_seq_mean_batch.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # print statistics
                tr_nvlb_loss_epoch_sum += tr_nvlb_seq_mean_batch.item()
                tr_kl_loss_epoch_sum += tr_kl_loss_seq_mean_batch.item()
                tr_nll_loss_epoch_sum -= tr_ll_loss_seq_mean_batch.item()

            scheduler.step()

            endtime = timer()
            # Measure wallclock time
            time_elapsed = endtime - starttime

            with torch.no_grad():
                
                for i, data in enumerate(val_loader, 0):
                    
                    val_Y_batch, val_X_batch = data
                    Y_val_batch = Variable(val_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                    val_latent_q_seq, val_mu_latent_q_seq, val_var_latent_q_seq, val_mu_latent_seq, val_var_latent_seq = model.forward(Y_val_batch)
                    val_mu_obs_given_latent_seq_curr, val_var_obs_given_latent_seq_curr = model.forward_emission_net(
                        latent_seq_current=val_latent_q_seq
                    )
                    val_kl_loss_seq_mean_batch, val_ll_loss_seq_mean_batch, val_nvlb_seq_mean_batch = dmm_loss_seq(
                        obs_seq=Y_val_batch,
                        mu_obs_seq=val_mu_obs_given_latent_seq_curr,
                        var_obs_seq=val_var_obs_given_latent_seq_curr,
                        mu_latent_q_seq=val_mu_latent_q_seq,
                        var_latent_q_seq=val_var_latent_q_seq,
                        mu_latent_seq=val_mu_latent_seq,
                        var_latent_seq=val_var_latent_seq
                    )
                    val_mse_loss_batch = mse_criterion(val_X_batch[:,:,:].to(device), val_mu_latent_q_seq)
                    # print statistics
                    # print statistics
                    val_nvlb_loss_epoch_sum += val_nvlb_seq_mean_batch.item()
                    val_kl_loss_epoch_sum += val_kl_loss_seq_mean_batch.item()
                    val_nll_loss_epoch_sum -= val_ll_loss_seq_mean_batch.item()
                    val_mse_loss_epoch_sum += val_mse_loss_batch.item()


            # Loss at the end of each epoch
            tr_nvlb_loss = tr_nvlb_loss_epoch_sum / len(train_loader)
            tr_nll_loss = tr_nll_loss_epoch_sum / len(train_loader)
            tr_kl_loss = tr_kl_loss_epoch_sum / len(train_loader)
            val_nvlb_loss = val_nvlb_loss_epoch_sum / len(val_loader)
            val_nll_loss = val_nll_loss_epoch_sum / len(val_loader)
            val_kl_loss = val_kl_loss_epoch_sum / len(val_loader)
            val_mse_loss = val_mse_loss_epoch_sum / len(val_loader)

            # Record the validation loss per epoch
            if (epoch + 1) > nepochs // 3: # nepochs/6 for complicated, 100 for simpler model
                model_monitor.record(val_nvlb_loss)

            # Displaying loss at an interval of 200 epochs
            if tr_verbose == True and (((epoch + 1) % 50) == 0 or epoch == 0):
                
                print("Epoch: {}/{}, Training: NLL:{:.4f}, KL :{:.4f}, NVLB:{:.4f}, Val: NLL:{:.4f}, KL :{:.4f}, NVLB:{:.4f}, MSE:{:.6f}, Time elapsed: {} s".format(epoch+1, 
                nepochs, tr_nll_loss, tr_kl_loss, tr_nvlb_loss, val_nll_loss, val_kl_loss, val_nvlb_loss, val_mse_loss, time_elapsed), file=orig_stdout)
                #save_model(model, model_filepath + "/" + "{}_ckpt_epoch_{}.pt".format(model.model_type, epoch+1))
                
                print("Epoch: {}/{}, Training: NLL:{:.4f}, KL :{:.4f}, NVLB:{:.4f}, Val: NLL:{:.4f}, KL :{:.4f}, NVLB:{:.4f}, MSE:{:.6f}, Time elapsed: {} s".format(epoch+1, 
                nepochs, tr_nll_loss, tr_kl_loss, tr_nvlb_loss, val_nll_loss, val_kl_loss, val_nvlb_loss, val_mse_loss, time_elapsed))
            
            # Checkpointing the model every few  epochs
            #if (((epoch + 1) % 500) == 0 or epoch == 0) and save_chkpoints == True:     
            if (((epoch + 1) % 10) == 0 or epoch == 0) and save_chkpoints == "all": 
                # Checkpointing model every few epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(model, model_filepath + "/" + "dmm_{}_{}_ckpt_epoch_{}.pt".format(model.inference_mode, model.rnn_type, epoch+1))
            elif (((epoch + 1) % nepochs) == 0) and save_chkpoints == "some": 
                # Checkpointing model at the end of training epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(model, model_filepath + "/" + "dmm_{}_{}_ckpt_epoch_{}.pt".format(model.inference_mode, model.rnn_type, epoch+1))
            
            # Save best model in case validation loss improves
            '''
            best_val_loss, best_model_wts, best_val_epoch, patience, check_patience = callback_val_loss(model=model,
                                                                                                    best_model_wts=best_model_wts,
                                                                                                    val_loss=val_loss,
                                                                                                    best_val_loss=best_val_loss,
                                                                                                    best_val_epoch=best_val_epoch,
                                                                                                    current_epoch=epoch+1,
                                                                                                    patience=patience,
                                                                                                    num_patience=num_patience,
                                                                                                    min_delta=min_delta,
                                                                                                    check_patience=check_patience,
                                                                                                    orig_stdout=orig_stdout)
            if check_patience == True:
                print("Monitoring validation loss for criterion", file=orig_stdout)
                print("Monitoring validation loss for criterion")
            else:
                pass
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss # Save best validation loss
                tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
                best_val_epoch = epoch+1 # Corresponding value of epoch
                best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model
            '''
            # Saving every value
            tr_nvlb_losses.append(tr_nvlb_loss)
            tr_kl_losses.append(tr_kl_loss)
            tr_nll_losses.append(tr_nll_loss)
            val_nvlb_losses.append(val_nvlb_loss)
            val_kl_losses.append(val_kl_loss)
            val_nll_losses.append(val_nll_loss)
            
            # Check monitor flag
            if model_monitor.monitor(epoch=epoch+1) == True:

                if tr_verbose == True:
                    print("Training convergence attained! Saving model at Epoch: {}".format(epoch+1), file=orig_stdout)
                
                print("Training convergence attained at Epoch: {}!".format(epoch+1))
                # Save the best model as per validation loss at the end
                best_val_loss = val_nvlb_loss # Save best validation loss
                tr_loss_for_best_val_loss = tr_nvlb_loss # Training loss corresponding to best validation loss
                best_val_epoch = epoch+1 # Corresponding value of epoch
                best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model
                #print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
                #save_model(model, model_filepath + "/" + "{}_usenorm_{}_ckpt_epoch_{}.pt".format(model.model_type, usenorm_flag, epoch+1))
                break

            #else:

                #print("Model improvement attained at Epoch: {}".format(epoch+1))
                #best_val_loss = val_loss # Save best validation loss
                #tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
                #best_val_epoch = epoch+1 # Corresponding value of epoch
                #best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model

            
        # Save the best model as per validation loss at the end
        print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
        
        #if save_chkpoints == True:
        if save_chkpoints == "all" or save_chkpoints == "some":
            # Save the best model using the designated filename
            if not best_model_wts is None:
                model_filename = "dmm_{}_{}_ckpt_epoch_{}_best.pt".format(model.inference_mode, model.rnn_type, best_val_epoch)
                torch.save(best_model_wts, model_filepath + "/" + model_filename)
            else:
                model_filename = "dmm_{}_{}_ckpt_epoch_{}_best.pt".format(model.inference_mode, model.rnn_type, epoch+1)
                print("Saving last model as best...")
                save_model(model, model_filepath + "/" + model_filename)
        #elif save_chkpoints == False:
        elif save_chkpoints == None:
            pass
    
    except KeyboardInterrupt:

        if tr_verbose == True:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1), file=orig_stdout)
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        else:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        
        if not save_chkpoints is None:
            model_filename = "dmm_{}_{}_ckpt_epoch_{}_latest.pt".format(model.inference_mode, model.rnn_type, epoch+1)
            torch.save(model, model_filepath + "/" + model_filename)

    print("------------------------------ Training ends --------------------------------- \n")
    # Restoring the original std out pointer
    sys.stdout = orig_stdout
    '''
    # Save a plot of training and validation losses
    plot_losses(tr_kl_loss_arr=np.array(tr_kl_losses), 
                tr_nll_loss_arr=np.array(tr_nll_losses),  
                tr_nvlb_loss_arr=np.array(tr_nvlb_losses), 
                val_kl_loss_arr=np.array(val_kl_losses), 
                val_nll_loss_arr=np.array(val_nll_losses), 
                val_nvlb_loss_arr=np.array(val_nvlb_losses), 
                savefig=True,
                savefig_name=os.path.join(os.path.split(logfile_path)[0], "loss_curves.pdf"))
    '''
    return tr_nvlb_losses, val_nvlb_losses, best_val_loss, tr_loss_for_best_val_loss, model
    


