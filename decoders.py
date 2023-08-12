import torch
from torch import nn
import torch.nn.functional as F


###########
# decoder #
###########
class decoder(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size, y_input_size, num_hidden_layers = 1, device = torch.device('cpu')):
        super(decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.y_input_size = y_input_size
        self.n_layers = num_hidden_layers
        self.device = device

        """
        N = batch size
        L = sequence length
        D = 2 if bidirectional=True otherwise 1
        H_in = input_size    
        H_cell = hidden_size
        H_out = proj_size if proj_size>0 otherwise hidden_size

        input = N x L x H_in    when  batch_first = True  else    L x N x H_in
        h_0 = D * num_layers x N x H_out
        c_0 = D * num_layers x  N x H_cell
        output = L x N x D * H_out  when batch_first = False    else    N x L x D * H_out
        """

        self.attn_layer = nn.Sequential( nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size) ,
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1)).to(device)

        self.lstm_layer = nn.LSTM(input_size = y_input_size, hidden_size = encoder_hidden_size, num_layers = num_hidden_layers, device=device)
        self.fc = nn.Linear(encoder_hidden_size + y_input_size, y_input_size, device=device)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, y_input_size, device=device)


    def init_hidden(self, x):
        return x.data.new(self.n_layers, x.size(0), self.decoder_hidden_size).zero_()


    def forward(self, input_encoded, y_history_last, fc_horizon, y_target, loss_func = None):

        if y_target == None:
            forecast_horizon = fc_horizon
        else:
            forecast_horizon = y_target.shape[1]

        input_window = input_encoded.shape[1]

        # input_encoded: batch_size x input_window x encoder_hidden_size

        # Initialize hidden and cell, 1 x batch_size x decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False

        y_pred_all = torch.zeros(input_encoded.shape[0], forecast_horizon, self.y_input_size).to(self.device)

        if len(y_history_last.shape) < 2: # If the number of dimensions(axes) is 1 (that is to say y is single-feature series then expand to 2 )
            y_prev = y_history_last.unsqueeze(1)
        else:
            y_prev = y_history_last

        loss = 0

        # print(next(self.attn_layer.parameters()).device)

        for t in range(forecast_horizon):

            # Use the last recurrent layer hidden states to calculate attention weights.
            last_layer_hidden = torch.unsqueeze(hidden[-1,:,:], 0) # 1 x batch_size x hidden_size
            last_layer_cell = torch.unsqueeze(cell[-1,:,:], 0) # 1 x batch_size x hidden_size

            # Eqn. 12-13: compute attention weights
            # batch_size x input_window x (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((last_layer_hidden.repeat(input_window, 1, 1).permute(1, 0, 2),
                           last_layer_cell.repeat(input_window, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            
            # x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size).shape = (batch_size * input_window) x (2*decoder_hidden_size + encoder_hidden_size)
            # self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).shape = (batch_size * input_window) x 1
            x = self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1, input_window)
            x = F.softmax(x, dim=1) # batch_size x input_window, row sum up to 1
            # Eqn. 14: compute context vector
            # torch.bmm(x.unsqueeze(1), input_encoded): [batch_size x 1 x input_window] * [batch_size x input_window x encoder_hidden_size] = [batch_size x 1 x encoder_hidden_size]
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size x encoder_hidden_size

            # Eqn. 15 without the concatenation of 'y' to the context
            #y_tilde = self.fc(context) # batch_size x 1

            y_tilde = self.fc(torch.cat((context, y_prev), dim = 1)) # batch_size x 1 (=y_input_size)

            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            # y_tilde.unsqueeze(0).shape = 1(=sequence length) x batch_size x 1 (=y_input_size)
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0] # N_layers x batch_size x decoder_hidden_size
            cell = lstm_output[1] # N_layers x batch_size x decoder_hidden_size

            # Eqn. 22: final output
            y_pred = self.fc_final(torch.cat((hidden[-1], context), dim = 1))
            y_pred_all[:,t,:] = y_pred

            if y_target == None:
                y_prev = y_pred
            else:
                y_prev = y_target[:, t] # teacher forcing

                if loss_func != None:
                    loss += loss_func(y_pred, y_target[:, t])

        return y_pred_all, loss

