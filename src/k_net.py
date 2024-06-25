# Adpoted from: https://github.com/KalmanNet/Unsupervised_EUSIPCO_22 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import gc
import sys
from utils.utils import count_params
from timeit import default_timer as timer

class KalmanNetNN(nn.Module):

    def __init__(self, n_states, n_obs, n_layers=1, device='cpu'):
        super(KalmanNetNN, self).__init__()

        self.n_states = n_states # Setting the number of states of the KalmanNet model
        self.n_obs = n_obs # Setting the number of observations of the KalmanNet model
        self.device = device # Setting the device
        
        # Setting the no. of neurons in hidden layers
        self.h1_knet = (self.n_states +  self.n_obs) * (10) * 8
        self.h2_knet = (self.n_states + self.n_obs) * (10) * 1
        self.d_in = self.n_states + self.n_obs # Input vector dimension for KNet 
        self.d_out = int(self.n_obs * self.n_states) # Output vector dimension for KNet

        # Setting the GRU specific nets
        self.input_dim = self.h1_knet # Input Dimension for the RNN
        self.hidden_dim = (self.n_states ** 2 + self.n_obs **2) * 10 # Hidden Dimension of the RNN
        self.n_layers = n_layers # Number of Layers in the GRU
        self.batch_size = 1 # Batch Size in the GRU
        self.seq_len_input = 1 # Input Sequence Length for the GRU
        self.seq_len_hidden = self.n_layers  # Hidden Sequence Length for the GRU (initilaized as the number of layers)
    
        # batch_first = False
        # dropout = 0.1 ;
        return None

    def Build(self, f, h):
        
        # Initialize the dynamics of the underlying ssm (equivalent of InitSystemDynamics(self, F, H) in original code)
        self.f_k = f
        self.h_k = h

        ##############################################
        # Initializing the Kalman Gain network
        # This network is: FC + RNN (e.g. GRU) + FC
        ##############################################
        # Input features are: 
            # 1. Innovation at time t: \delta y_t = y_t - \hat{y}_{t \vert t-1} (self.n_obs, 1) 
            # 2. Forward update difference: \delta x_{t-1} = \hat{x}_{t-1 \vert t-1} - \hat{x}_{t-1 \vert t-2} (self.n_states, 1)
        # Output of the net is the Kalman Gain computed at time t: K_t (self.n_states, self.n_obs)
        
        #############################
        ### Input Layer of KNet ###
        #############################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(self.d_in, self.h1_knet, bias=True).to(self.device, non_blocking = True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###################################
        ### RNN Network inside KNet ###
        ###################################
        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers,batch_first= True).to(self.device,non_blocking = True)

        #####################################
        ### Penultimate Layer of KNet ###
        #####################################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, self.h2_knet, bias=True).to(self.device,non_blocking = True)
        self.KG_relu2 = torch.nn.ReLU() # ReLU (Rectified Linear Unit) Activation Function

        #####################################
        ### Output Layer of KNet ###
        #####################################
        self.KG_l3 = torch.nn.Linear(self.h2_knet, self.d_out, bias=True).to(self.device,non_blocking = True)
        return None
    
    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):

        # Adjust for batch size
        M1_0 = torch.cat(self.batch_size*[M1_0],axis = 1).reshape(self.n_states, self.batch_size)
        self.m1x_prior = M1_0.detach().to(self.device, non_blocking = True) # Initial value of \mathbf{x}_{t \vert t-1}
        self.m1x_posterior = M1_0.detach().to(self.device, non_blocking = True) # Initial value of \mathbf{x}_{t-1 \vert t-1}
        self.state_process_posterior_0 = M1_0.detach().to(self.device, non_blocking = True)

    #########################################################
    ### Set Batch Size and initialize hidden state of GRU ###
    #########################################################
    def SetBatch(self,batch_size):
        self.batch_size = batch_size
        self.hn = torch.randn(self.seq_len_hidden,self.batch_size,self.hidden_dim,requires_grad=False).to(self.device)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.zeros_like(self.state_process_posterior_0).type(torch.FloatTensor).to(self.device)
        for i in range(self.state_process_posterior_0.shape[1]):
            #print(i, self.state_process_prior_0.shape, self.state_process_posterior_0.shape)
            self.state_process_prior_0[:, i] = self.f_k(self.state_process_posterior_0[:,i].reshape((-1,1))).view(-1,)
        #self.state_process_prior_0 = self.f_k(self.state_process_posterior_0) # torch.matmul(self.F,self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        #print(self.state_process_prior_0.shape, self.obs_process_0.shape)
        self.obs_process_0 = torch.zeros(self.n_obs, self.state_process_prior_0.shape[1]).type(torch.FloatTensor).to(self.device)
        #print(self.state_process_prior_0.shape, self.obs_process_0.shape)
        for i in range(self.state_process_posterior_0.shape[1]):
            #print(self.h_k(self.state_process_prior_0[:,i].reshape((-1,1))).view(-1,).shape)
            self.obs_process_0[:, i] = self.h_k(self.state_process_prior_0[:,i].reshape((-1,1))).view(-1,)
        #self.obs_process_0 = self.h_k(self.state_process_posterior_0) # torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.zeros_like(self.m1x_posterior).type(torch.FloatTensor).to(self.device)
        for i in range(self.m1x_posterior.shape[1]):
            self.m1x_prior[:,i] = self.f_k(self.m1x_posterior[:,i].reshape((-1,1))).view(-1,) # torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = torch.zeros(self.n_obs, self.m1x_prior.shape[1]).type(torch.FloatTensor).to(self.device)
        for i in range(self.m1y.shape[1]):
            self.m1y[:,i] = self.h_k(self.m1x_prior[:,i].reshape((-1,1))).view(-1,) # torch.matmul(self.H, self.m1x_prior)

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        
        self.step_prior() # Compute Priors

        self.step_KGain_est(y)  # Compute Kalman Gain

        dy = y - self.m1y # Compute the innovation
        
        #NOTE: My own change!!
        dy = func.normalize(dy, p=2, dim=0, eps=1e-12, out=None) # Extra normalization

        # Compute the 1-st posterior moment
        # Initialize array of Innovations
        INOV = torch.empty((self.n_states, self.batch_size),device= self.device)

        for batch in range(self.batch_size):
            # Calculate the Inovation for each KGain
            #print("batch: {}, KG norm: {}, dy norm: {}".format(batch+1, torch.norm(self.KGain[batch]).detach().cpu(), torch.norm(dy[:,batch]).detach().cpu()))
            INOV[:,batch] = torch.matmul(self.KGain[batch],dy[:,batch]).squeeze()
            assert torch.isnan(self.KGain[batch].detach().cpu()).any() == False, "NaNs in KG computation"
            assert torch.isnan(dy[:,batch].detach().cpu()).any() == False, "NaNs in innovation diff."

        self.m1x_posterior = self.m1x_prior + INOV

        del INOV,dy,y

        return torch.squeeze(self.m1x_posterior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        # dm1x = self.m1x_prior - self.state_process_prior_0 
        # (this is equivalent to Forward update difference: \delta x_{t-1} = \hat{x}_{t-1 \vert t-1} - \hat{x}_{t-1 \vert t-2} (self.n_states, 1))
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Normalize y 
        # (this is equivalent to Innovation at time t: \delta y_t = y_t - \hat{y}_{t \vert t-1} (self.n_obs, 1))
        dm1y = y - torch.squeeze(self.m1y) #y.squeeze() - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in.T)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.n_states, self.n_obs))
        del KG,KGainNet_in,dm1y,dm1x,dm1y_norm,dm1x_norm,dm1x_reshape


    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in)
        La1_out = self.KG_relu1(L1_out)
        assert torch.isnan(La1_out).any() == False, "NaNs in La1_out computation"

        ###########
        ### GRU ###
        ###########
        GRU_in = La1_out.reshape((self.batch_size,self.seq_len_input,self.input_dim))
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (self.batch_size, self.hidden_dim))
        assert torch.isnan(GRU_out_reshape).any() == False, "NaNs in GRU_output computation"

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)
        assert torch.isnan(La2_out).any() == False, "NaNs in La2_out computation"

        ####################
        ### Output Layer ###
        ####################
        self.L3_out = self.KG_l3(La2_out)
        assert torch.isnan(self.L3_out).any() == False, "NaNs in L3_out computation"

        del L2_out,La2_out,GRU_out,GRU_in,GRU_out_reshape,L1_out,La1_out
        return self.L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.T.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data

    def compute_predictions(self, y_test_batch):

        test_input = y_test_batch.to(self.device)
        N_T, Ty, dy = test_input.shape

        self.SetBatch(N_T)
        self.InitSequence(torch.zeros(self.ssModel.n_states, 1))

        x_out_test = torch.empty(N_T, self.n_states, Ty, device=self.device)
        y_out_test = torch.empty(N_T, self.n_obs, Ty, device=self.device)

        test_input = torch.transpose(test_input, 1, 2).type(torch.FloatTensor)

        for t in range(0, Ty):
            x_out_test[:,:, t] = self.forward(test_input[:,:, t]).T
            y_out_test[:,:, t] = self.m1y.T

        return x_out_test

def train_KalmanNetNN(model, options, train_loader, val_loader, nepochs,
                    logfile_path, modelfile_path, save_chkpoints, device='cpu', 
                    tr_verbose=False, unsupervised=True):
    
    # Set pipeline parameters: setting ssModel and model
    # 1. Set the KalmanNet model and push to its device
    model = model.to(device)

    # 2. Set the training parameters
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=options["lr"], 
                                 weight_decay=options["weight_decay"])
    
    loss_fn = nn.MSELoss(reduction='mean')
    MSE_cv_linear_epoch = np.empty([nepochs])
    MSE_cv_dB_epoch = np.empty([nepochs])

    MSE_train_linear_epoch = np.empty([nepochs])
    MSE_train_dB_epoch = np.empty([nepochs])

    MSE_cv_linear_epoch_obs = np.empty([nepochs])
    MSE_cv_dB_epoch_obs = np.empty([nepochs])

    MSE_train_linear_epoch_obs = np.empty([nepochs])
    MSE_train_dB_epoch_obs = np.empty([nepochs])

    # 3. Start the training and keep time statistics
    MSE_cv_dB_opt = 1000
    MSE_cv_idx_opt = 0

    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    #if save_chkpoints == True:
    if save_chkpoints == "all" or save_chkpoints == "some":
        # No grid search
        if logfile_path is None:
            training_logfile = "./log/knet_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    total_num_params, total_num_trainable_params = count_params(model)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))
    
    start_time = timer()
    for ti in range(0, nepochs):

        #################################
        ### Validation Sequence Batch ###
        #################################
        # Cross Validation Mode
        model.eval()

        # Load obserations and targets from CV data
        y_cv, cv_target = next(iter(val_loader))
        N_CV, Ty, dy = y_cv.shape
        #_, _, dx = cv_target.shape
        y_cv = torch.transpose(y_cv, 1, 2).type(torch.FloatTensor).to(model.device)
        cv_target = torch.transpose(cv_target, 1, 2).type(torch.FloatTensor).to(model.device)

        model.SetBatch(N_CV)
        model.InitSequence(torch.zeros(model.n_states,1))

        x_out_cv = torch.empty(N_CV, model.ssModel.n_states, Ty, device= device).to(model.device)
        y_out_cv = torch.empty(N_CV, model.ssModel.n_obs, Ty, device= device).to(model.device)

        for t in range(0, Ty):
            #print("Time instant t:{}".format(t+1))
            x_out_cv[:,:, t] = model(y_cv[:,:, t]).T
            y_out_cv[:,:,t] = model.m1y.squeeze().T

        # Compute Training Loss
        cv_loss = loss_fn(x_out_cv[:,:,:Ty], cv_target[:,:,:]).item()
        cv_loss_obs =  loss_fn(y_out_cv[:,:,:Ty], y_cv[:,:,:Ty]).item()

        # Average
        MSE_cv_linear_epoch[ti] = np.mean(cv_loss)
        MSE_cv_dB_epoch[ti] = 10 * np.log10(MSE_cv_linear_epoch[ti])

        MSE_cv_linear_epoch_obs[ti] = np.mean(cv_loss_obs)
        MSE_cv_dB_epoch_obs[ti] = 10*np.log10(MSE_cv_linear_epoch_obs[ti])

        relevant_loss = cv_loss_obs if unsupervised else cv_loss
        relevant_loss = 10 * np.log10(relevant_loss)

        if (relevant_loss < MSE_cv_dB_opt):
            MSE_cv_dB_opt = relevant_loss
            MSE_cv_idx_opt = ti
            print("Saving model ...")
            torch.save(model.state_dict(), modelfile_path + "/" + "knet_ckpt_epoch_best.pt")

        ###############################
        ### Training Sequence Batch ###
        ###############################

        # Training Mode
        model.train()

        # Init Hidden State
        model.init_hidden()

        Batch_Optimizing_LOSS_sum = 0

        # Load random batch sized data, creating new iter ensures the data is shuffled
        y_training, train_target = next(iter(train_loader))
        N_E, Ty, dy = y_training.shape
        y_training = torch.transpose(y_training, 1, 2).type(torch.FloatTensor).to(model.device)
        train_target = torch.transpose(train_target, 1, 2).type(torch.FloatTensor).to(model.device)

        model.SetBatch(N_E)
        model.InitSequence(torch.zeros(model.n_states,1))

        x_out_training = torch.empty(N_E, model.n_states, Ty, device=device).to(model.device)
        y_out_training = torch.empty(N_E, model.n_obs, Ty,device=device).to(model.device)

        for t in range(0, Ty):
            x_out_training[:,:,t] = model(y_training[:,:,t]).T
            y_out_training[:,:,t] = model.m1y.squeeze().T

        # Compute Training Loss
        loss  = loss_fn(x_out_training[:,:,:Ty], train_target[:,:,:])
        loss_obs  = loss_fn(y_out_training[:,:,:Ty], y_training[:,:,:Ty])

        # Select loss, from which to update the gradient
        LOSS = loss_obs if unsupervised else loss

        # Average
        MSE_train_linear_epoch[ti] = loss
        MSE_train_dB_epoch[ti] = 10 * np.log10(MSE_train_linear_epoch[ti])

        MSE_train_linear_epoch_obs[ti] = loss_obs
        MSE_train_dB_epoch_obs[ti] = 10*np.log10(MSE_train_linear_epoch_obs[ti])

        ##################
        ### Optimizing ###
        ##################

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        # Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
        LOSS.backward()
        
        #NOTE; Try to use gradient clipping with l2-norm of 1.0 (arbitrary choice) to deal with longer trajectories
        #nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        ########################
        ### Training Summary ###
        ########################
        time_elapsed = timer() - start_time
        train_print = MSE_train_dB_epoch_obs[ti] if unsupervised else MSE_train_dB_epoch[ti]
        cv_print = MSE_cv_dB_epoch_obs[ti] if unsupervised else MSE_cv_dB_epoch[ti]
        print(ti, "MSE Training :", train_print, "[dB]", "MSE Validation :", cv_print,"[dB]", "Time elapsed :", time_elapsed)
        print(ti, "MSE Training :", train_print, "[dB]", "MSE Validation :", cv_print,"[dB]", "Time elapsed :", time_elapsed, file=orig_stdout)

        if (ti > 1):
            d_train = MSE_train_dB_epoch_obs[ti] - MSE_train_dB_epoch_obs[ti - 1] if unsupervised \
                    else MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]


            d_cv = MSE_cv_dB_epoch_obs[ti] - MSE_cv_dB_epoch_obs[ti - 1] if unsupervised \
                    else MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]

            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")
            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]", file=orig_stdout)
            

        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")
        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]", file=orig_stdout)

        # reset hidden state gradient
        model.hn.detach_()

        # Reset the optimizer for faster convergence
        if ti % 50 == 0 and ti != 0:
            optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=options["lr"], 
                                 weight_decay=options["weight_decay"])
            print('Optimizer has been reset')
            print('Optimizer has been reset', file=orig_stdout)

    sys.stdout = orig_stdout
    return MSE_train_dB_epoch_obs, MSE_cv_dB_epoch_obs, model

def test_KalmanNetNN(model_test, test_loader, options, device, model_file=None, test_logfile_path = None):

    with torch.no_grad():

        N_T = options["N_T"]
        # Load test data and create iterator
        test_data_iter = iter(test_loader)

        # Allocate Array
        MSE_test_linear_arr = torch.empty([N_T],device = device)
        MSE_test_linear_arr_obs = torch.empty([N_T],device= device)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='none')
        # Set model in evaluation mode
        model_test.load_state_dict(torch.load(model_file))
        model_test = model_test.to(device)
        model_test.eval()
        
        # Load training data from iter
        test_input,test_target = next(test_data_iter)
        test_input = torch.transpose(test_input, 1, 2)
        test_target = torch.transpose(test_target, 1, 2)
        test_target = test_target.to(device)
        test_input = test_input.to(device)
        _, Ty, dy = test_input.shape

        model_test.SetBatch(N_T)
        model_test.InitSequence(model_test.ssModel.m1x_0)

        if not test_logfile_path is None:
            test_log = "./log/test_danse.log"
        else:
            test_log = test_logfile_path
    
        x_out_test = torch.empty(N_T, model_test.n_states, Ty, device=device)
        y_out_test = torch.empty(N_T, model_test.n_obs, Ty, device=device)

        for t in range(0, Ty):
            x_out_test[:,:, t] = model_test(test_input[:,:, t]).T
            y_out_test[:,:, t] = model_test.m1y.T

        loss_unreduced = loss_fn(x_out_test[:,:,:Ty],test_target[:,:,:Ty])
        loss_unreduced_obs = loss_fn(y_out_test[:,:,:Ty],test_input[:,:,:Ty])

        # Create the linear loss from the total loss for the batch
        loss = torch.mean(loss_unreduced,axis = (1,2))
        loss_obs = torch.mean(loss_unreduced_obs,axis = (1,2))


        MSE_test_linear_arr[:] = loss
        MSE_test_linear_arr_obs[:] = loss_obs

        # Average
        MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
        MSE_test_dB_avg = 10 * torch.log10(MSE_test_linear_avg).item()

        MSE_test_linear_avg_obs = torch.mean(MSE_test_linear_arr_obs)
        MSE_test_dB_avg_obs = 10 * torch.log10(MSE_test_linear_avg_obs).item()

        with open(test_log, "a") as logfile_test:
            logfile_test.write('Test MSE loss: {:.3f}, Test MSE loss obs: {:.3f} using weights from file: {}'.format(MSE_test_dB_avg, MSE_test_dB_avg_obs, model_file))
        # Print MSE Cross Validation
        #str = self.modelName + "-" + "MSE Test:"
        #print(str, self.MSE_test_dB_avg, "[dB]")
    
    return MSE_test_dB_avg, MSE_test_dB_avg_obs, x_out_test
