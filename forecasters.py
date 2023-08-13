from numpy import isin
import torch
import torch.nn as nn
import numpy as np
import sys

from utils import unwindow_series, window_series, fold_indices, random_permute_indices, unfold, all_metric_lists, append_all_metrics, mean_metrics
from model_utils import get_best_device, activ_func_from_type, reset_weights, optimizer_from_type


class MultiStepForecaster(nn.Module):

  def __init__(self):

    super(MultiStepForecaster, self).__init__()

    self.loss = nn.MSELoss()
    self.device = get_best_device()

  def get_device(self):
    return self.device


  def train_cross_validation(self, X_folds, y_folds, config, verbose=True, extra_data = None):

    fld_range = range(len(X_folds))

    losses = []
    all_metrics = all_metric_lists()

    y, fold_offsets = unfold(y_folds)
    
    if extra_data != None:
      extra_data['predictions'] = {}

    for valid_fld in fld_range:

      reset_weights(self, verbose)

      # Training

      train_flds = [ifld for ifld in fld_range if ifld != valid_fld]

      X_train =  torch.from_numpy(np.vstack([X_folds[ifld] for ifld in train_flds])).type(torch.FloatTensor).to(self.get_device())
      y_train =  torch.from_numpy(np.vstack([y_folds[ifld] for ifld in train_flds])).type(torch.FloatTensor).to(self.get_device())

      rand_perm_idx = random_permute_indices(X_train.shape[0])
      X_train_rand = X_train[rand_perm_idx,:,:]
      y_train_rand = y_train[rand_perm_idx,:]

      self.train(X_train_rand, y_train_rand, config, verbose = verbose)

      # Validation

      X_valid = X_folds[valid_fld]
      y_valid = y_folds[valid_fld]

      # Create non-overlapping windows
      X_valid_unwin = unwindow_series(X_valid) # We assume that window_step = 1 initially
      y_valid_unwin = unwindow_series(y_valid) # We assume that window_step = 1 initially

      step_no_overlap = y_valid.shape[1]

      X_valid_no_overlap_win = window_series(X_valid_unwin, X_valid.shape[1], 0, step_no_overlap )
      y_valid_no_overlap_win = window_series(y_valid_unwin, y_valid.shape[1], 0, step_no_overlap )

      X_valid = torch.from_numpy(X_valid_no_overlap_win).type(torch.FloatTensor).to(self.get_device())
      y_valid = torch.from_numpy(y_valid_no_overlap_win).type(torch.FloatTensor).to(self.get_device())

      y_valid_pred, valid_loss = self.test(X_valid, y_valid)
      losses.append(valid_loss.item())

      window_size = X_valid.shape[1]

      append_all_metrics(all_metrics, y_valid.detach().cpu().numpy(), y_valid_pred.detach().cpu().numpy())

      if verbose:
        print('Fold {} Validation loss {}'.format(valid_fld, valid_loss))

      if extra_data != None:
        y_valid_pred = y_valid_pred.detach().cpu().numpy()
        y_valid_pred_unwin = unwindow_series(y_valid_pred, y_valid_pred.shape[1])
        extra_data['predictions'][fold_offsets[valid_fld]+window_size] = y_valid_pred_unwin

    mean_loss = np.mean(losses)
    print('Mean fold validation loss {}'.format(mean_loss))

    all_mean_metrics = mean_metrics(all_metrics)
    print('Mean fold Metrics: ' + str(all_mean_metrics))

    return mean_loss



  def train(self, X_train, y_train, config, verbose = True):

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    optimizer_type = config['optimizer_type']

    l = self.loss

    optimizer = optimizer_from_type(optimizer_type, self, learning_rate)

    epoch_losses = []

    for epoch in range(num_epochs):

      batch_losses = []

      batch_start = 0
      y_pred = torch.empty(y_train.shape, device=self.get_device())
      
      while batch_start < X_train.shape[0]:

        batch_end = batch_start+batch_size
        X_train_batch = X_train[batch_start:batch_end]
        y_train_batch = y_train[batch_start:batch_end]

        #forward feed
        y_pred_batch = self(X_train_batch, forecast_horizon = y_train_batch.shape[1])

        #calculate the loss
        loss = l(y_pred_batch, y_train_batch) # The loss will usually be average over all samples in the batch. reduction parameter to the criterion decides whether to average or sum them

        y_pred[batch_start:batch_end,:] = y_pred_batch
        
        batch_start = batch_end

        #backward propagation: calculate gradients
        loss.backward()

        #update the weights
        optimizer.step()

        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        batch_losses.append(loss.item())

      epoch_loss = np.mean(batch_losses)

      epoch_losses.append(epoch_loss.item())

      if verbose:
        if epoch % (num_epochs // 10) == 0:
          print('epoch {}, loss {}'.format(epoch, epoch_loss), end='\r')

    return y_pred, epoch_losses



  def test(self, X_test, y_test):

    l = self.loss

    y_pred = self(X_test, forecast_horizon = y_test.shape[1])
    loss = l(y_pred, y_test)

    return y_pred, loss



class S2SForecaster(MultiStepForecaster):

  def __init__(self, input_size=1, output_size=1, hidden_size = 64, hidden_layers = 1):

    super(S2SForecaster, self).__init__()
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

    self.num_layers = hidden_layers
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = self.num_layers, device=self.device)
    self.decoder = nn.LSTM(input_size = output_size, hidden_size = hidden_size, num_layers = self.num_layers, device=self.device)
    self.fc = nn.Linear(hidden_size, output_size, device=self.device)


  def trainable_parameters(self):
    params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
    params += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
    params += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
    return params


  def forward(self, x, forecast_horizon = 1, y=None):

    # x: batch_size x input_window x input_size
    # y: batch_size x forecast_horizon x output_size

    batch_size = x.shape[0]

    # Encoder

    enc_hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device) # N_layers x batch_size x hidden_size (proj_size = 0 - if proj_size>0 it would be: N_layers * batch_size x proj_size)
    enc_cell = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device) # N_layers x batch_size x hidden_size
  
    x_seq_first = x.permute(1, 0, 2) # batch_first = false

    _, (enc_hidden, enc_cell) = self.encoder(x_seq_first, (enc_hidden, enc_cell))

    # Decoder

    dec_hidden = enc_hidden
    dec_cell = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device) # N_layers x batch_size x hidden_size

    y_out = torch.empty(batch_size, forecast_horizon, self.output_size, device=self.device)

    if y == None:

      y_out_t = torch.zeros(1, batch_size, self.output_size, device=self.device)

      for t in range(forecast_horizon):
        out_t, (dec_hidden, dec_cell) = self.decoder(y_out_t, (dec_hidden, dec_cell))
        y_out_t = self.fc(out_t)
        y_out[:,t,:] = y_out_t

    else: # Use ground-truth y to produce the forecasts

      y_seq_first = y.permute(1, 0, 2) # batch_first = false
      out,_ = self.decoder(y_seq_first, (dec_hidden, dec_cell))

      forecast_horizon = y.shape[1]-1 # The 1st ground-truth y of the forecast window is overlapping with the last x of the input window.

      for t in range(forecast_horizon):
        y_out[:,t,:] = self.fc(out[t,:,:]) # drop last forecast

    return y_out

