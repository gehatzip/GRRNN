import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np


from importlib import reload
from rcgan import RCGAN, LSRCGAN, train_iteration_RCGAN, cgan_cond, noise_like_sequence
from wrcgan import WRCGAN, train_iteration_WRCGAN
from forecasters import MultiStepForecaster
import encoders
import decoders

from utils import unwindow_series, window_series, random_permute_indices, unfold, all_metric_lists, append_all_metrics, mean_metrics

reload(encoders)
reload(decoders)

from model_utils import optimizer_from_type, reset_weights

##########
# da_rnn #
##########
class da_rnn:
    
    def __init__(self, gs_input_size, fs_input_size, input_window
                    , learning_rate = 0.01, generative_learning = True
                    , cell_dimension = 64, num_hidden_layers = 1, dropout=0.0, L1_coeff = 1.0, gan_disc_decay=0.0, device = torch.device('cpu')):

        # gs_input_size: Number of features of the generating series.
        # fs_input_size: Number of features of the series to be forecasted.

        self.fs_input_size = fs_input_size # Forecast series: Number of features
        self.input_window = input_window # Input window is stored as parameter in the DARNN because the encoder attention layer is sized by that.
        self.generative_learning = generative_learning # Cut-off RCGAN if false
        self.gan_disc_decay = gan_disc_decay

        self.decoder = decoders.decoder(encoder_hidden_size = cell_dimension,
                            decoder_hidden_size = cell_dimension,
                            y_input_size = fs_input_size, num_hidden_layers = num_hidden_layers, device=device)

        if generative_learning:
            self.encoder = encoders.encoder(input_size = cell_dimension, hidden_size = cell_dimension
                                , input_window = input_window, num_hidden_layers = num_hidden_layers, device = device)
        else:
            self.encoder = encoders.encoder(input_size = gs_input_size, hidden_size = cell_dimension
                                , input_window = input_window, num_hidden_layers = num_hidden_layers, device = device)

        # GAN input_size = X_input_size + y_input_size + noise_size (chosen equal to: y_input_size)
        # GAN hidden_size = X_input_size + y_input_size
        
        self.gan = WRCGAN(input_size = fs_input_size, cond_in_size = gs_input_size, noise_size = fs_input_size
                          , hidden_size = cell_dimension, num_hidden_layers = num_hidden_layers, dropout = dropout, L1_coeff = L1_coeff, device = device)

        # Module-level parallelism
        """
        self.gan = nn.DataParallel(self.gan)
        self.encoder = nn.DataParallel(self.encoder) # Implements data parallelism at the module level. This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension (other objects will be copied once per device).
        self.decoder = nn.DataParallel(self.decoder)
        """

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
        
        """
        This function is used for testing.
        'y_history' can be removed worsening the performance. 
        If so then the windowing of the target series for testing only made inside 'prepare_windowed_series()' can be the same as models without Generative branch!
        """

        input_window = X.shape[1]

        if self.generative_learning:

            # y_history is the condition of the CGAN.
            if len(y_history.shape) < 2: # If y_history is a single-feature series expand to two dimensions
                y_history = y_history.unsqueeze(2) # batch_size x input_window x n_forecast_features

            noise = noise_like_sequence(y_history.shape, X.device)
            cond = cgan_cond(X)

            gen_input = torch.cat((cond, noise), dim=2) # batch_size x input_window x (fs_input_size + noise_input_size(=fs_input_size) )
            ## gen_input = cond # batch_size x input_window x fs_input_size
            gen_y_history, gen_hidden, gen_cell = self.gan.G(gen_input) 
            # gen_y_history: batch_size x input_window x n_forecast_features
            # gen_hidden: n_layers x batch_size x input_window x hidden_dim

            # Concatenate the GAN-generated y_history with X and pass to the encoder
            # encoder_input = torch.cat((X, gen_hidden[-1,:,:,:]), dim=2)
            # encoder_input = torch.cat((y_history, gen_hidden[-1,:,:,:]), dim=2)
            encoder_input = gen_hidden[-1,:,:,:]
            
        else:
            encoder_input = X

        _, decoder_input = self.encoder(encoder_input)

        # If RCGAN generated y_target then we would pass it to the decoder in the case of generative learning.
        gen_y_target = None

        y_history_last = y_history[:,-1,:]
        # y_history_last = torch.zeros(y_history.shape[0], y_history.shape[2], device=self.device)

        outp,_ = self.decoder(decoder_input, y_history_last, forecast_horizon, gen_y_target)

        if self.generative_learning:
            gen_y = gen_y_history
        else:
            gen_y = None

        return outp, gen_y

 

    def train_iteration(self, X, gen_hidden, y_history, y_target, criterion):

        # X: batch_size x window_size x N_X_features
        # gen_hidden: batch_size x window_size x hidden_dim

        if self.generative_learning:
            # encoder_input = torch.cat((X, gen_hidden), dim=2)
            # encoder_input = torch.cat((y_history, gen_hidden), dim=2)
            encoder_input = gen_hidden
            
        else:
            encoder_input = X

        _, decoder_input = self.encoder(encoder_input)

        y_history_last = y_history[:,-1,:]
        # y_history_last = torch.zeros(y_history.shape[0], y_history.shape[2], device=self.device)

        y_pred, loss = self.decoder(input_encoded = decoder_input, y_history_last = y_history_last, fc_horizon = y_target.shape[1], y_target = y_target, loss_func = criterion)
        
        return y_pred, loss



    def train_batch(self, X, y_history, y_target, encoder_optimizer, decoder_optimizer, gan_gen_optimizer=None, gan_disc_optimizer=None, gen_loss_weight=0.5):

        loss_func = nn.MSELoss()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        if self.generative_learning:
            
            gen_loss, disc_loss, gen_hidden, _, gen_y_history = train_iteration_WRCGAN(self.gan, X, y_history, gan_gen_optimizer, gan_disc_optimizer) # gen_y_history.shape = batch_size x input_window x n_forecast_features
            y_pred, enc_dec_loss = self.train_iteration(X, gen_hidden[-1,:,:,:], y_history, y_target, loss_func) # encoder decoder loss
            
            loss = (1-gen_loss_weight)*enc_dec_loss + gen_loss_weight*gen_loss
            loss.backward()
            gan_gen_optimizer.step()
            
        else:
            disc_loss = None
            gen_loss = None
            enc_dec_loss = None
            gen_y_history = None
            y_pred, loss = self.train_iteration(X, None, y_history, y_target, loss_func) # encoder decoder loss
            loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return y_pred, loss, gen_loss, disc_loss, gen_y_history





class DARNNForecaster(MultiStepForecaster):

  def __init__(self, input_size=1, output_size=1, hidden_size = 64, hidden_layers = 1, window_size=6, dropout=0.0, L1_coeff = 1.0, gan_disc_decay=0.0, generative = True):

    super(DARNNForecaster, self).__init__()

    self.darnn = da_rnn(gs_input_size=input_size, fs_input_size=output_size, input_window = window_size
                    , cell_dimension = hidden_size, num_hidden_layers = hidden_layers, dropout = dropout
                    , L1_coeff = L1_coeff, gan_disc_decay=gan_disc_decay, generative_learning = generative, device = self.device)

  def trainable_parameters(self):
    return self.darnn.trainable_parameters()


  # Overrides 'train' of super-class
  def train(self, X_train, y_train, config, verbose = True):

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    optimizer_type = config['optimizer_type']
    gan_disc_learning_rate = config['gan_disc_learning_rate']
    gen_loss_weight = config['gen_loss_weight']

    window_size = X_train.shape[1]

    encoder_optimizer = optimizer_from_type(optimizer_type, self.darnn.encoder, learning_rate)
    decoder_optimizer = optimizer_from_type(optimizer_type, self.darnn.decoder, learning_rate)
    gan_gen_optimizer = optimizer_from_type(optimizer_type, self.darnn.gan.G, learning_rate)
    gan_disc_optimizer = optimizer_from_type(optimizer_type, self.darnn.gan.D, gan_disc_learning_rate, self.darnn.gan_disc_decay)

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

      if self.darnn.generative_learning:
        y_gen = torch.empty((y_train.shape[0], window_size, y_train.shape[2]), device=self.get_device())
      else:
        y_gen = None;
      
      while batch_start < X_train.shape[0]:

        batch_end = batch_start+batch_size
        X_train_batch = X_train[batch_start:batch_end]
        y_train_batch = y_train[batch_start:batch_end]

        y_pred_batch, loss, gen_loss, disc_loss, y_gen_batch = self.darnn.train_batch(X_train_batch, y_train_batch[:,:window_size,:], y_train_batch[:,window_size:,:], encoder_optimizer, decoder_optimizer, gan_gen_optimizer, gan_disc_optimizer, gen_loss_weight)

        y_pred[batch_start:batch_end,:] = y_pred_batch

        batch_losses.append(loss.item())

        if self.darnn.generative_learning:
          y_gen[batch_start:batch_end,:] = y_gen_batch
          batch_gen_losses.append(gen_loss.item())
          batch_disc_losses.append(disc_loss.item())

        batch_start = batch_end

      epoch_loss = np.mean(batch_losses)

      if self.darnn.generative_learning:
        epoch_gen_loss = np.mean(batch_gen_losses)
        epoch_disc_loss = np.mean(batch_disc_losses)

      epoch_losses.append(epoch_loss.item())

      """
      # Debugging:
      if verbose:
        if epoch % (num_epochs // 10) == 0:
          if self.darnn.generative_learning:
            print('epoch {}, loss {}, gen_loss {}, disc_loss {}'.format(epoch, epoch_loss, epoch_gen_loss, epoch_disc_loss), end='\r\r')
            plot_real_vs_gen(y_train[:,:window_size,:], y_gen)
          else:
            print('epoch {}, loss {}'.format(epoch, epoch_loss), end='\r')
      """

    return y_pred, epoch_losses


  # Overrides 'test' of super-class
  def test(self, X_test, y_test):

    l = self.loss

    window_size = X_test.shape[1]
    y_history = y_test[:,:window_size,:]
    forecast_horizon = y_test.shape[1]-window_size
    y_pred,_ = self.darnn.predict_overlapping(X_test, y_history, forecast_horizon)
    
    y_target = y_test[:,window_size:,:]

    loss = l(y_pred, y_target)

    return y_pred, loss

  # Overrides 'train_cross_validation' of super-class
  def train_cross_validation(self, X_folds, y_folds, config, verbose=True, extra_data = None):

    fld_range = range(len(X_folds))

    losses = []
    all_metrics = all_metric_lists()

    y, fold_offsets = unfold(y_folds)
    
    if extra_data != None:
      extra_data['predictions'] = {}

    for valid_fld in fld_range:

      self.darnn.reset_weights(verbose)

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
