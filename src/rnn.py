import numpy as np
import scipy
import sys
import torch
from torch import nn
import torch.nn.functional as F

# Create an RNN model for prediction
class RNN_model(nn.Module):
    """ This super class defines the specific model to be used i.e. LSTM or GRU or RNN
    """
    def __init__(self, input_size, output_size, n_hidden, n_layers, 
        model_type, lr, num_epochs, n_hidden_dense=32, num_directions=1, batch_first = True, min_delta=1e-2):
        super(RNN_model, self).__init__()
        """
        Args:
        - input_size: The dimensionality of the input data
        - output_size: The dimensionality of the output data
        - n_hidden: The size of the hidden layer, i.e. the number of hidden units used
        - n_layers: The number of hidden layers
        - model_type: The type of modle used ("lstm"/"gru"/"rnn")
        - lr: Learning rate used for training
        - num_epochs: The number of epochs used for training
        - num_directions: Parameter for bi-directional RNNs (usually set to 1 in this case, for bidirectional set as 2)
        - batch_first: Option to have batches with the batch dimension as the starting dimension 
        of the input data
        """
        # Defining some parameters
        self.hidden_dim = n_hidden  
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        
        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        
        # Predefined:
        ## Use only the forward direction 
        self.num_directions = num_directions
        
        ## The input tensors must have shape (batch_size,...)
        self.batch_first = batch_first
        
        # Defining the recurrent layers 
        if model_type.lower() == "rnn": # RNN 
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)   
        elif model_type.lower() == "lstm": # LSTM
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "gru": # GRU
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)  
        else:
            print("Model type unknown:", model_type.lower()) 
            sys.exit() 
        
        # Fully connected layer to be used for mapping the output
        #self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)
        
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size)
        # Add a dropout layer with 20% probability
        #self.d1 = nn.Dropout(p=0.2)

    def init_h0(self, batch_size):
        """ This function defines the initial hidden state of the RNN
        """
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0
    
    def forward(self, x):
        """ This function defines the forward function to be used for the RNN model
        """
        batch_size = x.shape[0]
        
        # Obtain the RNN output
        r_out, hn_all = self.rnn(x)
        
        # Reshaping the output appropriately
        r_out = r_out.contiguous().view(batch_size, -1, self.num_directions, self.hidden_dim)
        
        # Select the last time-step
        r_out = r_out[:, -1, :, :]
        r_out_last_step = r_out.reshape((-1, self.hidden_dim))
        
        # Pass this through dropout layer
        #r_out_last_step = self.d1(r_out_last_step)

        # Passing the output to the fully connected layer
        y = F.relu(self.fc(r_out_last_step))

        # Means and variances are computed       
        mu = self.fc_mean(y)
        vars = F.relu(self.fc_vars(y))

        return mu, vars

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    return None

def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets