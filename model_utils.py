import torch
import torch.nn as nn
import numpy as np


def get_best_device():

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  
  # device = torch.device('cpu')

  return device


def activ_func_from_type(act_type):
  if act_type == 'Sigmoid':
    return nn.Sigmoid()
  elif act_type == 'LeakyReLU':
    return nn.LeakyReLU()
  elif act_type == 'Tanh':
    return nn.Tanh()
  else:
    return nn.ReLU()


def reset_weights(model, verbose=True):

    for name, module in model.named_children():

        if hasattr(module, 'reset_parameters'):

            if verbose:
              print('resetting ', name)
              
            module.reset_parameters()


def optimizer_from_type(optimizer_type, model, learning_rate, weight_decay=0.0):

  if hasattr(model, '__iter__'):
    parameters = []
    for submodel in model:
      parameters += list(submodel.parameters())
  else:
      parameters = model.parameters()

  if optimizer_type == 'Adam':
    return torch.optim.Adam(parameters, lr =learning_rate, weight_decay = weight_decay)
  elif optimizer_type == 'Adagrad':
    return torch.optim.Adagrad(parameters, lr =learning_rate, weight_decay = weight_decay)
  else: # SGD by default
    return torch.optim.SGD(parameters, lr =learning_rate, weight_decay = weight_decay)



def loss_for_model(model_type):

  if model_type == 'forecast':

    return nn.MSELoss()
  else:
    return nn.CrossEntropyLoss()

