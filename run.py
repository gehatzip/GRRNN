import pandas as pd
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import read_ETD, read_SML2010, read_AirQuality, read_EnergyCo, read_poll_all_parties
from smac_optim import smac_cross_validation, cross_validation_config, smac_cross_validation_optimize, train_test_config, smac_config_from_file, smac_train_validation_testing_optimize, smac_initial_config
from arg_handlers import horizon_cmd_arg, dataset_cmd_arg

from series import normalize

import sys


def forecast_and_target_series(dataset, project_root_dir = ''):
        
    forecast_ts = None
    target_ts = None

    if dataset == 'SML2010':
        dset = read_SML2010(project_root_dir)
        target_cols = ['3:Temperature_Comedor_Sensor','4:Temperature_Habitacion_Sensor']
    elif dataset == 'AirQuality':
        dset = read_AirQuality(project_root_dir)
        target_cols = ['CO(GT)','NMHC(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']
    elif dataset == 'energyco':
        dset = read_EnergyCo(project_root_dir)
        target_cols = ['Appliances','lights']
    elif dataset == 'poll':
        dset = read_poll_all_parties(project_root_dir)
        target_cols = ['Dem_poll', 'Rep_poll']
    else:
        dset = read_ETD()
        target_cols = ['OT']

    all_cols = dset.columns[:]
    forecast_cols = [col for col in all_cols if col not in target_cols]

    forecast_ts = dset[forecast_cols].astype(np.float32).to_numpy()
    target_ts = dset[target_cols].astype(np.float32).to_numpy()

    return forecast_ts, target_ts


def main():
    
    usage_msg = '\nUsage:\
                \n --dataset [ETD|SML2010|AirQuality|energyco|poll]\
                \n --horizon: [N]\
                '

    if len(sys.argv) < 2 or '--help' in sys.argv:
        print(usage_msg)
        sys.exit()

    forecast_horizon = horizon_cmd_arg()

    dataset = dataset_cmd_arg()

    torch.manual_seed(0)
    np.random.seed(0)

    forecast_ts, target_ts = forecast_and_target_series(dataset)

    predict_type = 'forecast'

    # Cross validation with SMAC-optimized hyperparameters
    """
    smac_cross_validation(forecast_ts, target_ts, forecast_horizon)
    """
    
    """
    smac_train_validation_testing_optimize(forecast_ts, target_ts, forecast_horizon)
    """

    # Cross-validation-ptimized configurarion
    
    """
    # Preprocessing
    forecast_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    forecast_ts_norm = normalize(forecast_ts, forecast_scaler)
    target_ts_norm = normalize(target_ts, target_scaler)
    config = smac_cross_validation_optimize(forecast_ts_norm, target_ts_norm, forecast_horizon)
    """

    # Initial configuration
    """
    config = smac_initial_config()
    """

    # Configuration from file
    config = smac_config_from_file('initial_configuration.txt')
    

    # Cross validation with configuration
    """
    cross_validation_config(config, forecast_ts, target_ts, forecast_horizon)
    """


    # Train-test with coniguration
    train_test_config(config, forecast_ts, target_ts, forecast_horizon)


    



if __name__ == "__main__":
    main()
