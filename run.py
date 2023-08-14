import pandas as pd
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import read_ETD, read_SML2010, read_AirQuality, read_EnergyCo, read_poll_all_parties, series_split
from smac_optim import smac_config_from_file, smac_config_to_file, smac_train_validation_optimize, smac_train, series_preprocess, smac_test, smac_plot
from arg_handlers import horizon_cmd_arg, dataset_cmd_arg, mode_cmd_arg

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
                \n --mode [optimize|train|test]\
                \n --horizon: [N]\
                '

    if len(sys.argv) < 2 or '--help' in sys.argv:
        print(usage_msg)
        sys.exit()

    forecast_horizon = horizon_cmd_arg()

    dataset = dataset_cmd_arg()

    mode = mode_cmd_arg()

    torch.manual_seed(0)
    np.random.seed(0)

    forecast_ts, target_ts = forecast_and_target_series(dataset)

    target_scaler = MinMaxScaler()

    forecast_ts_norm, target_ts_norm = series_preprocess(forecast_ts, target_ts, MinMaxScaler(), target_scaler)

    forecast_ts_train_valid, target_ts_train_valid, forecast_ts_test, target_ts_test = series_split(forecast_ts_norm, target_ts_norm, ratio=0.8)
    forecast_ts_train, target_ts_train, forecast_ts_valid, target_ts_valid = series_split(forecast_ts_train_valid, target_ts_train_valid, ratio=0.8)
    
    if mode == 'optimize':
        smac_train_validation_optimize(forecast_ts_train, target_ts_train, forecast_ts_valid, target_ts_valid, forecast_horizon)
    elif mode == 'train':
        optimized_config = smac_config_from_file('optimized_configuration.txt')
        smac_train(optimized_config, forecast_ts_train_valid, target_ts_train_valid, forecast_horizon, verbose = False, model_save_file = 'model.pt')
    elif mode == 'test':
        extra_data = {}
        optimized_config = smac_config_from_file('optimized_configuration.txt')
        smac_test(optimized_config, forecast_ts_test, target_ts_test, forecast_horizon, verbose = False, extra_data = extra_data)
        smac_plot(optimized_config, target_ts, target_ts_train_valid, target_scaler, extra_data)




if __name__ == "__main__":
    main()
