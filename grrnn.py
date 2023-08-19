import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np


from importlib import reload
from wrcgan import WRCGAN, train_iteration_WRCGAN, cgan_cond, noise_like_sequence
from forecasters import MultiStepForecaster
import encoders
import decoders

from utils import unwindow_series, window_series, random_permute_indices, unfold, all_metric_lists, append_all_metrics, mean_metrics, plot_windowed

reload(encoders)
reload(decoders)

from model_utils import optimizer_from_type, reset_weights

##########
# gr_rnn #
##########
class gr_rnn:
    
    def __init__(self, gs_input_size, fs_input_size, input_window
                    , learning_rate = 0.01, cell_dimension = 64, num_hidden_layers = 1
                    , device = torch.device('cpu')):
        
        self.fs_input_size = fs_input_size
        self.input_window = input_window

        self.decoder = decoders.decoder(encoder_hidden_size = cell_dimension,
                            decoder_hidden_size = cell_dimension,
                            y_input_size = fs_input_size, num_hidden_layers = num_hidden_layers, device=device)

        self.encoder = encoders.encoder(input_size = cell_dimension, hidden_size = cell_dimension
                              , input_window = input_window, num_hidden_layers = num_hidden_layers, device = device)
        
        self.gan = WRCGAN(input_size = fs_input_size, cond_in_size = gs_input_size, noise_size = fs_input_size
                          , hidden_size = cell_dimension, num_hidden_layers = num_hidden_layers, device = device)

        self.figsize=(18, 6)
        self.dpi=80

        self.device = device


    def device(self):
        return self.device

    def reset_weights(self, verbose):
        reset_weights(self.encoder, verbose)
        reset_weights(self.decoder, verbose)
        self.gan.reset_weights(verbose)

    def trainable_parameters(self):
      params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
      params += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
      params += sum(p.numel() for p in self.gan.parameters() if p.requires_grad)
      return params

    def predict_overlapping(self, X, y_history, forecast_horizon):

        input_window = X.shape[1]

        # y_history is the condition of the CGAN.
        if len(y_history.shape) < 2: # If y_history is a single-feature series expand to two dimensions
            y_history = y_history.unsqueeze(2) # batch_size x input_window x n_forecast_features

        noise = noise_like_sequence(y_history.shape, X.device)
        cond = cgan_cond(X)

        gen_input = torch.cat((cond, noise), dim=2)
        gen_y_history, gen_hidden, gen_cell = self.gan.G(gen_input) 
        encoder_input = gen_hidden[-1,:,:,:]

        _, decoder_input = self.encoder(encoder_input)

        # If RCGAN generated y_target then we would pass it to the decoder in the case of generative learning.
        gen_y_target = None

        y_history_last = y_history[:,-1,:]

        outp,_ = self.decoder(decoder_input, y_history_last, forecast_horizon, gen_y_target)

        gen_y = gen_y_history

        return outp, gen_y

 

    def train_iteration(self, X, gen_hidden, y_history_last, y_target, criterion):

        encoder_input = gen_hidden

        _, decoder_input = self.encoder(encoder_input)

        y_pred, loss = self.decoder(input_encoded = decoder_input, y_history_last = y_history_last, fc_horizon = y_target.shape[1], y_target = y_target, loss_func = criterion)
        
        return y_pred, loss



    def train_batch(self, X, y_history, y_target, encoder_optimizer, decoder_optimizer, gan_gen_optimizer=None, gan_disc_optimizer=None):

        loss_func = nn.MSELoss()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        y_history_last = y_history[:,-1,:]

        gen_loss, disc_loss, gen_hidden, _, gen_y_history = train_iteration_WRCGAN(self.gan, X, y_history, gan_gen_optimizer, gan_disc_optimizer) # gen_y_history.shape = batch_size x input_window x n_forecast_features
        y_pred, enc_dec_loss = self.train_iteration(X, gen_hidden[-1,:,:,:], y_history_last, y_target, loss_func) # encoder decoder loss
        
        loss = enc_dec_loss + gen_loss
        loss.backward()
        gan_gen_optimizer.step()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return y_pred, loss, gen_loss, disc_loss, gen_y_history


class GRRNNForecaster(MultiStepForecaster):

  def __init__(self, input_size=1, output_size=1, hidden_size = 64, hidden_layers = 1, window_size=6):

    super(GRRNNForecaster, self).__init__()

    self.grrnn = gr_rnn(gs_input_size=input_size, fs_input_size=output_size, input_window = window_size
                    , cell_dimension = hidden_size, num_hidden_layers = hidden_layers, device = self.device)

  def trainable_parameters(self):
    return self.grrnn.trainable_parameters()


  # Overrides 'train' of super-class
  def train(self, X_train, y_train, batch_size = 16, num_epochs = 100, learning_rate = 0.01, optimizer_type = 'SGD', verbose = True):

    window_size = X_train.shape[1]

    encoder_optimizer = optimizer_from_type(optimizer_type, self.grrnn.encoder, learning_rate)
    decoder_optimizer = optimizer_from_type(optimizer_type, self.grrnn.decoder, learning_rate)
    gan_gen_optimizer = optimizer_from_type(optimizer_type, self.grrnn.gan.G, learning_rate)
    gan_disc_optimizer = optimizer_from_type(optimizer_type, self.grrnn.gan.D, learning_rate)
    

    epoch_losses = []
    epoch_gen_losses = []
    epoch_enc_dec_losses = []

    for epoch in range(num_epochs):

      batch_losses = []
      batch_gen_losses = []
      batch_disc_losses = []

      batch_start = 0
      forecast_horizon = y_train.shape[1]-window_size
      y_pred = torch.empty((y_train.shape[0], forecast_horizon, y_train.shape[2]), device=self.get_device())

      y_gen = torch.empty((y_train.shape[0], window_size, y_train.shape[2]), device=self.get_device())
      
      while batch_start < X_train.shape[0]:

        batch_end = batch_start+batch_size
        X_train_batch = X_train[batch_start:batch_end]
        y_train_batch = y_train[batch_start:batch_end]

        y_pred_batch, loss, gen_loss, disc_loss, y_gen_batch = self.grrnn.train_batch(X_train_batch, y_train_batch[:,:window_size,:], y_train_batch[:,window_size:,:]
                                                                                      , encoder_optimizer, decoder_optimizer, gan_gen_optimizer, gan_disc_optimizer)

        y_pred[batch_start:batch_end,:] = y_pred_batch

        batch_losses.append(loss.item())

        y_gen[batch_start:batch_end,:] = y_gen_batch
        batch_gen_losses.append(gen_loss.item())
        batch_disc_losses.append(disc_loss.item())

        batch_start = batch_end

      epoch_loss = np.mean(batch_losses)

      epoch_gen_loss = np.mean(batch_gen_losses)
      epoch_disc_loss = np.mean(batch_disc_losses)

      epoch_losses.append(epoch_loss.item())

    return y_pred, epoch_losses


  # Overrides 'test' of super-class
  def test(self, X_test, y_test):

    l = self.loss

    window_size = X_test.shape[1]
    y_history = y_test[:,:window_size,:]
    forecast_horizon = y_test.shape[1]-window_size
    y_pred,_ = self.grrnn.predict_overlapping(X_test, y_history, forecast_horizon)
    
    y_target = y_test[:,window_size:,:]

    loss = l(y_pred, y_target)

    return y_pred, loss


  # Overrides 'train_cross_validation' of super-class
  def train_cross_validation(self, X_folds, y_folds, batch_size = 16, num_epochs = 100, learning_rate = 0.01, optimizer_type = 'SGD', verbose=True, extra_data = None):

    fld_range = range(len(X_folds))

    losses = []
    all_metrics = all_metric_lists()

    y, fold_offsets = unfold(y_folds)
    
    if extra_data != None:
      extra_data['predictions'] = {}

    for valid_fld in fld_range:

      self.grrnn.reset_weights(verbose)

      # Training

      train_flds = [ifld for ifld in fld_range if ifld != valid_fld]

      X_train =  torch.from_numpy(np.vstack([X_folds[ifld] for ifld in train_flds])).type(torch.FloatTensor).to(self.get_device())
      y_train =  torch.from_numpy(np.vstack([y_folds[ifld] for ifld in train_flds])).type(torch.FloatTensor).to(self.get_device())

      rand_perm_idx = random_permute_indices(X_train.shape[0])
      X_train_rand = X_train[rand_perm_idx,:,:]
      y_train_rand = y_train[rand_perm_idx,:]

      self.train(X_train_rand, y_train_rand, batch_size, num_epochs, learning_rate, verbose = verbose)

      # Validation

      X_valid = X_folds[valid_fld]
      y_valid = y_folds[valid_fld]

      # Create non-overlapping windows
      X_valid_unwin = unwindow_series(X_valid) # We assume that window_step = 1 initially
      y_valid_unwin = unwindow_series(y_valid) # We assume that window_step = 1 initially

      step_no_overlap = y_valid.shape[1]-X_valid.shape[1]

      X_valid_no_overlap_win = window_series(X_valid_unwin, X_valid.shape[1], 0, step_no_overlap )
      y_valid_no_overlap_win = window_series(y_valid_unwin, y_valid.shape[1], 0, step_no_overlap )

      X_valid = torch.from_numpy(X_valid_no_overlap_win).type(torch.FloatTensor).to(self.get_device())
      y_valid = torch.from_numpy(y_valid_no_overlap_win).type(torch.FloatTensor).to(self.get_device())

      y_valid_pred, valid_loss = self.test(X_valid, y_valid)
      losses.append(valid_loss.item())


      window_size = X_valid.shape[1]
      y_valid_target = y_valid[:,window_size:,:]

      append_all_metrics(all_metrics, y_valid_target.detach().cpu().numpy(), y_valid_pred.detach().cpu().numpy())

      if verbose:
        print('Fold {} Validation loss {}'.format(valid_fld, valid_loss))

      if extra_data != None:
        y_valid_pred = y_valid_pred.detach().cpu().numpy()
        y_valid_pred_unwin = unwindow_series(y_valid_pred, y_valid_pred.shape[1])
        extra_data['predictions'][fold_offsets[valid_fld]+window_size] = y_valid_pred_unwin

    mean_loss = np.mean(losses)
    print('Mean fold validation loss {}'.format(mean_loss))

    all_mean_metrics = mean_metrics(all_metrics)
    print('Mean fold Metrics: ' +str(all_mean_metrics))

    return mean_loss
