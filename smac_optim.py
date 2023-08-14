from re import L
import numpy as np
import torch

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, Constant, UniformFloatHyperparameter, CategoricalHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB  
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from grrnn import GRRNNForecaster

from utils import error_metric_func, window_series, unwindow_series, fold, series_split, plot_real_vs_predicted, calc_all_error_metrics

from sklearn.preprocessing import MinMaxScaler
from series import normalize, denormalize, denormalize_fold_dict

from math import isinf
from sys import float_info
from os import linesep

class SMACModelCrossValidRunner:

    def __init__(self, forecast_ts, target_ts, forecast_horizon, tae):
        self.forecast_ts = forecast_ts
        self.target_ts = target_ts
        self.tae = tae
        self.forecast_horizon = forecast_horizon

    def run(self, config):
        return self.tae(config, self.forecast_ts, self.target_ts, self.forecast_horizon, False)


class SMACModelTrainTestRunner:

    def __init__(self, forecast_ts_train, target_ts_train, forecast_ts_test, target_ts_test, forecast_horizon, tae):
        self.forecast_ts_train = forecast_ts_train
        self.target_ts_train = target_ts_train
        self.forecast_ts_test = forecast_ts_test
        self.target_ts_test = target_ts_test
        self.tae = tae
        self.forecast_horizon = forecast_horizon

    def run(self, config):
        return self.tae(config, self.forecast_ts_train, self.target_ts_train, self.forecast_ts_test, self.target_ts_test, self.forecast_horizon, False)


def get_initial_config_space():

    configspace = ConfigurationSpace()

    configspace.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 0.01, 0.05, default_value=0.02))
    configspace.add_hyperparameter(UniformIntegerHyperparameter('batch_size', 16, 32, default_value=24))
    configspace.add_hyperparameter(UniformIntegerHyperparameter('num_epochs', 10, 70, default_value=50))
    
    configspace.add_hyperparameter(UniformIntegerHyperparameter('hidden_dim', 24, 64, default_value=48))

    # configspace.add_hyperparameter(Constant('window_size', 7))
    configspace.add_hyperparameter(UniformIntegerHyperparameter('window_size', 7, 14, default_value=10))

    """
    # 'hidden_layers' is excluded as hyper-parameters because the comparable models have RNNs with a single hidden layer
    configspace.add_hyperparameter(UniformIntegerHyperparameter('hidden_layers', 1, 4, default_value=1))
    """
    configspace.add_hyperparameter(Constant('hidden_layers', 1))

    # configspace.add_hyperparameter(UniformFloatHyperparameter('gen_loss_weight', 0.1, 0.9, default_value=0.5))
    configspace.add_hyperparameter(Constant('gen_loss_weight', 0.5))
    configspace.add_hyperparameter(UniformFloatHyperparameter('gan_disc_learning_rate', 0.01, 0.05, 0.02))
    configspace.add_hyperparameter(UniformFloatHyperparameter('dropout', 0.01, 0.4, default_value=0.2))
    configspace.add_hyperparameter(UniformFloatHyperparameter('L1_coeff', 0.1, 1.0, default_value=0.5))
    configspace.add_hyperparameter(UniformFloatHyperparameter('gan_disc_decay', 0.0, 0.1, default_value=0.01))

    """
    configspace.add_hyperparameter(CategoricalHyperparameter('optimizer', choices=['SGD', 'Adam', 'Adagrad'], default_value='Adam'))
    """
    configspace.add_hyperparameter(Constant('optimizer', 'Adam'))

    return configspace


def create_model(window_size, forecast_horizon, input_size, output_size, hidden_size, hidden_layers, activ_func=None, dropout=0.0, L1_coeff=1.0, gan_disc_decay=0.0):

    model = GRRNNForecaster(input_size, output_size, hidden_size, hidden_layers, window_size, dropout, L1_coeff, gan_disc_decay)

    return model


def prepare_windowed_series(forecast_ts, target_ts, window_size, forecast_horizon, window_step = 1):
    
    X = window_series(forecast_ts, window_size, 0, window_step) # Windows of input features: N x window_size x N_features

    y = window_series(target_ts, window_size+forecast_horizon, 0, window_step) # Windows of output features: N x window_size+forecast_horizon x N_forecast_features
    
    if not ((X is None) or (y is None)) and (X.shape[0] > y.shape[0]):
        X = X[:y.shape[0],:,:]

    return X, y


def config_cross_validation_train(config, forecast_ts, target_ts, forecast_horizon, verbose = False, extra_data = None):

    #############
    X, y = prepare_windowed_series(forecast_ts, target_ts, int(config['window_size']), forecast_horizon)

    # Cross-validation folds
    K_folds=4
    X_folds = fold(X, K_folds) 
    y_folds = fold(y, K_folds)
    #############

    activ_func = None

    if 'activ_func' in config:
        activ_func = config['activ_func']

    model = create_model(X_folds[0].shape[1], forecast_horizon, X_folds[0].shape[2], y_folds[0].shape[-1]
                         , int(config['hidden_dim']), config['hidden_layers'], activ_func, config['dropout'], config['L1_coeff'], config['gan_disc_decay'])

    return model.train_cross_validation(X_folds, y_folds, config, verbose=verbose, extra_data = extra_data)


def smac_scenario():

    configspace = get_initial_config_space()

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 100,  # Max number of function evaluations (the more the better)
        "cs": configspace
    })

    return scenario


def smac_train_test_optimize(forecast_ts_train, target_ts_train, forecast_ts_test, target_ts_test, forecast_horizon):

    scenario = smac_scenario()

    tae_runner_owner = SMACModelTrainTestRunner(forecast_ts_train, target_ts_train, forecast_ts_test, target_ts_test, forecast_horizon, smac_train_test)

    smac = SMAC4HPO(scenario=scenario, tae_runner=tae_runner_owner.run)

    best_config = smac.optimize()

    return best_config


def smac_config_to_file(config, file = 'optimized_configuration.txt'):

    f = open(file, 'w')

    for hparam in config.items():
        f.write(hparam[0] + ' ' + str(hparam[1]) + linesep)

    f.close()


def series_preprocess(forecast_ts, target_ts, forecast_scaler, target_scaler):
    # Preprocess
    forecast_ts_norm = normalize(forecast_ts, forecast_scaler)
    target_ts_norm = normalize(target_ts, target_scaler)

    return forecast_ts_norm, target_ts_norm


def series_train_valid_test_split(forecast_ts, target_ts):
    # train, validation and test split
    forecast_ts_train_valid, target_ts_train_valid, forecast_ts_test, target_ts_test = series_split(forecast_ts_norm, target_ts_norm, ratio=0.8)
    forecast_ts_train, target_ts_train, forecast_ts_valid, target_ts_valid = series_split(forecast_ts_train_valid, target_ts_train_valid, ratio=0.8)

    return forecast_ts_train, target_ts_train, forecast_ts_valid, target_ts_valid, forecast_ts_test, target_ts_test


def smac_train_validation_optimize(forecast_ts_train, target_ts_train, forecast_ts_valid, target_ts_valid, forecast_horizon):

    optimized_config = smac_train_test_optimize(forecast_ts_train, target_ts_train, forecast_ts_valid, target_ts_valid, forecast_horizon)

    print('\n Optimized configuration: \n')
    print(optimized_config)

    smac_config_to_file(optimized_config)



def hyperparams_from_file(filename):

    f = open(filename, 'r')

    config = {}

    with open(filename, 'r') as f:

        lines = f.readlines()

        for line in lines: 
            
            words = line.strip().split()

            if len(words) > 1:
                config[words[0]] = words[1]
                print('{}: {}'.format(words[0], words[1]))

    return config


def smac_config_from_file(filename):
    hyperparams_dict = hyperparams_from_file(filename)
    return smac_config_from_dict(hyperparams_dict)


def smac_config_from_dict(cfg_dict):

    configspace = get_initial_config_space()
    config = configspace.sample_configuration()

    config['learning_rate'] = float(cfg_dict['learning_rate'])
    config['batch_size'] = int(cfg_dict['batch_size'])
    config['num_epochs'] = int(cfg_dict['num_epochs'])

    if 'hidden_dim' in config.keys():
        config['hidden_dim'] = int(cfg_dict['hidden_dim'])

    optimizer_hp = configspace.get_hyperparameter('optimizer')
    if optimizer_hp.is_legal(cfg_dict['optimizer']):
        config['optimizer'] = cfg_dict['optimizer']

    config['window_size'] = int(cfg_dict['window_size'])

    if 'hidden_layers' in config.keys():
        config['hidden_layers'] = int(cfg_dict['hidden_layers'])

    if 'gen_loss_weight' in config.keys():
        config['gen_loss_weight'] = float(cfg_dict['gen_loss_weight'])

    if 'gan_disc_learning_rate' in config.keys():
        config['gan_disc_learning_rate'] = float(cfg_dict['gan_disc_learning_rate'])

    if 'dropout' in config.keys():
        config['dropout'] = float(cfg_dict['dropout'])

    if 'L1_coeff' in config.keys():
        config['L1_coeff'] = float(cfg_dict['L1_coeff'])

    if 'gan_disc_decay' in config.keys():
        config['gan_disc_decay'] = float(cfg_dict['gan_disc_decay'])

    return config


def mean_metric(all_metrics):
    res = 0

    for val in all_metrics.values():
        if isinf(val) == False:
            res += val

    res = res / len(all_metrics)
    return res

def smac_train_test(config, forecast_ts_train, target_ts_train, forecast_ts_test, target_ts_test, forecast_horizon, verbose = False, extra_data = None):

    activ_func = None

    if 'activ_func' in config:
        activ_func = config['activ_func']

    window_size = int(config['window_size'])

    X_train, y_train = prepare_windowed_series(forecast_ts_train, target_ts_train, window_size, forecast_horizon)

    X_test, y_test = prepare_windowed_series(forecast_ts_test, target_ts_test, window_size, forecast_horizon, forecast_horizon) # By using forecast_horizon as step in test windowed series the target windows will be non-overlapping!

    if (X_train is None) or (y_train is None) or (X_test is None) or (y_test is None):
        print('Empty windowed series - invalid train/test!')
        return float_info.max

    hidden_layers = config['hidden_layers']

    model = create_model(X_train.shape[1], forecast_horizon, X_train.shape[2], y_train.shape[-1]
                         , config['hidden_dim'], hidden_layers, activ_func, config['dropout'], config['L1_coeff'], config['gan_disc_decay'])

    if extra_data is not None:
        extra_data['model'] = model

    if verbose:
        print('Trainable parameters: ' + str(model.trainable_parameters()))

    X_train_tensor = torch.from_numpy(X_train).type(torch.FloatTensor).to(model.get_device())
    y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor).to(model.get_device())

    
    model.train(X_train_tensor, y_train_tensor, config, verbose)

    X_test_tensor = torch.from_numpy(X_test).type(torch.FloatTensor).to(model.get_device())
    y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor).to(model.get_device())

    y_pred, loss = model.test(X_test_tensor, y_test_tensor)

    window_size = X_test.shape[1]
    y_target = y_test[:,window_size:,:]

    y_pred_np = y_pred.detach().cpu().numpy()

    all_metrics = calc_all_error_metrics(y_target, y_pred_np)

    print('Test loss {}'.format(loss))
    print('Error metrics: ' + str(all_metrics))

    if extra_data != None:
      if y_pred_np.ndim < 3:
        y_pred_np = np.expand_dims(y_pred_np, axis=1) # n_samples x forecast_horizon(=1) x n_features

      y_pred_unwin = unwindow_series(y_pred_np, forecast_horizon)
      extra_data['predictions'] = y_pred_unwin

    # return loss.item()
    return mean_metric(all_metrics)



def smac_train(config, forecast_ts_train, target_ts_train, forecast_horizon, verbose = False, model_save_file = 'model.pt'):

    activ_func = None

    if 'activ_func' in config:
        activ_func = config['activ_func']

    window_size = int(config['window_size'])

    X_train, y_train = prepare_windowed_series(forecast_ts_train, target_ts_train, window_size, forecast_horizon)

    if (X_train is None) or (y_train is None):
        print('Empty windowed series - invalid train series!')
        return float_info.max

    hidden_layers = config['hidden_layers']

    model = create_model(X_train.shape[1], forecast_horizon, X_train.shape[2], y_train.shape[-1]
                         , config['hidden_dim'], hidden_layers, activ_func, config['dropout'], config['L1_coeff'], config['gan_disc_decay'])

    if verbose:
        print('Trainable parameters: ' + str(model.trainable_parameters()))

    X_train_tensor = torch.from_numpy(X_train).type(torch.FloatTensor).to(model.get_device())
    y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor).to(model.get_device())

    
    model.train(X_train_tensor, y_train_tensor, config, verbose)

    torch.save(model, model_save_file)


def smac_test(config, forecast_ts_test, target_ts_test, forecast_horizon, verbose = False, extra_data = None, model_save_file = 'model.pt'):

    window_size = int(config['window_size'])

    X_test, y_test = prepare_windowed_series(forecast_ts_test, target_ts_test, window_size, forecast_horizon, forecast_horizon) # By using forecast_horizon as step in test windowed series the target windows will be non-overlapping!

    if (X_test is None) or (y_test is None):
        print('Empty windowed series - invalid test series!')
        return float_info.max

    model = torch.load(model_save_file)

    X_test_tensor = torch.from_numpy(X_test).type(torch.FloatTensor).to(model.get_device())
    y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor).to(model.get_device())

    y_pred, loss = model.test(X_test_tensor, y_test_tensor)

    window_size = X_test.shape[1]
    y_target = y_test[:,window_size:,:]

    y_pred_np = y_pred.detach().cpu().numpy()

    all_metrics = calc_all_error_metrics(y_target, y_pred_np)

    print('Test loss {}'.format(loss))
    print('Error metrics: ' + str(all_metrics))

    if extra_data != None:
      if y_pred_np.ndim < 3:
        y_pred_np = np.expand_dims(y_pred_np, axis=1) # n_samples x forecast_horizon(=1) x n_features

      y_pred_unwin = unwindow_series(y_pred_np, forecast_horizon)
      extra_data['predictions'] = y_pred_unwin

    # return loss.item()
    return mean_metric(all_metrics)


def smac_plot(optimized_config, target_ts, target_ts_train_valid, target_scaler, extra_data):

    last_target_ts_train_valid = target_ts[ target_ts_train_valid.shape[0]-1, :]
    target_ts_pred = denormalize(extra_data['predictions'], target_scaler, last_target_ts_train_valid)

    pred_start = target_ts_train_valid.shape[0] + int(optimized_config['window_size'])

    target_ts_aligned = target_ts[pred_start:pred_start+target_ts_pred.shape[0], :]

    all_metrics = calc_all_error_metrics(target_ts_aligned, target_ts_pred)
    print('Initial series error metrics: ' + str(all_metrics))
    
    plot_real_vs_predicted(target_ts, target_ts_pred, pred_start)
