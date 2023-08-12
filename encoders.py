import torch
from torch import nn
import torch.nn.functional as F

###########
# encoder #
###########
class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, input_window, num_hidden_layers = 1, device = torch.device('cpu')):
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_window = input_window
        self.n_layers = num_hidden_layers

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_hidden_layers, device=device)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + input_window, out_features = 1, device=device)


    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return x.data.new(self.n_layers, x.size(0), self.hidden_size).zero_() # dimension 1 is the batch dimension



    def forward(self, input_data):

        # input_data: batch_size x input_window x input_size        
        input_weighted = input_data.data.new(input_data.size(0), self.input_window, self.input_size).zero_() # batch_size x input_window x input_size
        input_encoded = input_data.data.new(input_data.size(0), self.input_window, self.hidden_size).zero_() # # batch_size x input_window x hidden_size

        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # N_layers * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False

        for t in range(self.input_window):
            
            # input_data.shape = input_weighted.shape = input_encoded.shape = batch_size x input_window x input_size
            # hidden.shape = N_layers x batch_size x hidden_size
            # last_layer_hidden.repeat(self.input_size, 1, 1).shape = input_size x batch_size x hidden_size
            # last_layer_hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2).shape = batch_size x input_size x hidden_size
            # input_data.permute(0, 2, 1).shape = batch_size x input_size x input_window
            # x.shape = batch_size x input_size x (2*hidden_size + input_window)
            
            # Use the last recurrent layer hidden states to calculate attention weights.
            last_layer_hidden = torch.unsqueeze(hidden[-1,:,:], 0) # 1 x batch_size x hidden_size
            last_layer_cell = torch.unsqueeze(cell[-1,:,:], 0) # 1 x batch_size x hidden_size

            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((last_layer_hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2), # last_layer_hidden.repeat(self.input_size, 1, 1) : input_size x batch_size x hidden_size after permute: batch_size x input_size x hidden_size
                           last_layer_cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), # batch_size x input_size x input_window
                           dim = 2) # batch_size x input_size x (2*hidden_size + input_window)
            
            # tensor.repeat(m, n): create a new tensor that contains mxn copies of the initial ordered in m rows and n columns
            
            # Eqn. 9: Get attention weights
            # x.view(-1, self.hidden_size * 2 + self.input_window).shape = (batch_size * input_size) x (2*self.hidden_size + self.input_window)
            x_view = x.view(-1, 2*self.hidden_size + self.input_window)

            x = self.attn_linear(x_view) # (batch_size * input_size) x 1
            
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=1) # batch_size x input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size x input_size
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters() # Stores the parameters as a contiguous memory piece
            
            # weighted_input.unsqueeze(0).shape = 1(=sequence_size) x batch_size x input_size
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell)) # '_,' denotes missing variables for skipped return values
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden[-1,:,:] # cell state is not used - only last layer of hidden state!
            
        return input_weighted, input_encoded

