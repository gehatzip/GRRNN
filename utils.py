from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

def normalize(df):

  scaler = MinMaxScaler()
  df_norm = scaler.fit_transform(df)
  df_norm = pd.DataFrame(df_norm, columns=df.columns, index=df.index)

  return scaler, df_norm


def window_series(series, window_size, offset = 0, step = 1):
    # Converts series (N_time x N_features) to series of windows (N_windows x window_size x N_features)
    # Windows start from the 'offset' given and shift by 'step'.

    n_time_steps = series.shape[0]
    windows = []
    win_start = offset

    win_end = offset + window_size

    if win_end > n_time_steps:
      print('ERROR: Series could not be windowed: offset/input_window({}) + horizon({}) > time samples({})'.format(offset, window_size, n_time_steps))

    while win_end <= n_time_steps:
        win_start = win_end - window_size
        windows.append(series[win_start:win_end,:])
        win_end = win_end+step

    windows_arr = None

    if len(windows) > 0:

        windows_arr = np.empty((len(windows), windows[0].shape[0], windows[0].shape[1]))
        
        for i, window in enumerate(windows):
            windows_arr[i,:,:] = window

    return windows_arr


def unwindow_series(series_win, step=1): 
    # series_win: Windowed series (n_windows x window_size [x n_features])
    # step is the predefined shift of the initial series between two samples with the same position in two consecutive windows

    window_size = series_win.shape[1]

    if series_win.ndim < 3:
        series_win = np.expand_dims(series_win, axis=2)

    series = np.zeros((window_size+(series_win.shape[0]-1)*step, series_win.shape[2]))

    for i in range(0, series_win.shape[0]):
        iserial = i*step
        series[iserial:iserial+window_size,:] = series_win[i,:,:]

    return series

def plot_windowed(x_win, step=1, offset=0):
  x = unwindow_series(x_win, step)
  plt.plot(range(1+offset, 1 + offset + x.shape[0]), x)


def fold_indices(rows, folds):

    rem = rows % folds
    step = rows // folds
    step = step + 2*rem//folds
    idx = np.arange(step,rows,step)

    if len(idx) >= folds:
        idx = idx[0:folds-1]

    return idx


def random_permute_indices(rows):
    return np.random.permutation(rows)

def MAE(y_true, y_pred): 
    # Mean Absolute Error
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def MAPE(y_true, y_pred):
    # Mean Absolute Percentage Error
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def SMAPE(y_true, y_pred): 
    # Symmetric Mean Absolute Percentage Error (handles zero values)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(2*(y_true - y_pred) / (np.abs(y_true)+np.abs(y_pred))))

def RMSE(y_true, y_pred):
    # Root Mean Square Error (handles zero values)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))


def error_metric_func(err_type='SMAPE'):
  
  err_func = SMAPE
  if err_type == 'MAPE':
    err_func = MAPE
  elif err_type == 'MAE':
    err_func = MAE
  elif err_type == 'RMSE':
    err_func = RMSE

  return err_func



def all_metric_types():
  return ['SMAPE', 'MAPE', 'MAE', 'RMSE']


def all_metric_lists():

  metric_types = all_metric_types()
  all_metrics = {}

  for metric_type in metric_types:
    all_metrics[metric_type] = []
  
  return all_metrics

def append_all_metrics(all_metrics, y_truth, y_pred):

  new_metrics = {}

  for metric_type, metric_list in all_metrics.items():
    metric_func = error_metric_func(metric_type)
    new_metric = metric_func(y_truth, y_pred)
    metric_list.append(new_metric)
    new_metrics[metric_type] = new_metric

  return new_metrics


def mean_metrics(metric_lists):
  mean_metrics = {}
  for type, metric_list in metric_lists.items():
    mean_metrics[type] = np.mean(metric_list)
  return mean_metrics


def calc_all_error_metrics(y_target, y_pred):

  metric_types = all_metric_types()
  all_metrics = {}
  for metric_name in metric_types:
    err_metric_cb = error_metric_func(metric_name)
    all_metrics[metric_name] = err_metric_cb(y_target, y_pred)

  return all_metrics


def plot_real_vs_predicted_feature(ax, real, predicted, predicted_offset = 0, feature_name = 'Feature 0', x_labels = None):

  t = np.arange(real.shape[0])

  if x_labels:
    ax.plot(x_labels, real, label='Real')
    ax.tick_params(axis='x', rotation=45, labelsize=6)
  else:
    ax.plot(t, real, label='Real')
    
  t_pred = t+predicted_offset
  t_pred_offset = t_pred[:predicted.shape[0]]
  ax.plot(t_pred_offset, predicted, label='Predicted')

  ymin = min(real.min(), predicted.min())
  ymax = max(real.max(), predicted.max())
  stdev = ymax - ymin

  ax.set_ylim(([ymin-stdev, ymax+stdev]))
  ax.set_title(feature_name)
  
  ax.legend()


def plot_real_vs_predicted(real, predicted, predicted_offset = 0, feature_names = None):

  n_features = predicted.shape[1]

  if n_features > 1:

    fig, axs = plt.subplots(1, n_features, figsize = (16,8))

    for i in range(n_features):

      if feature_names != None and len(feature_names) > i:
        feature_name = feature_names[i]
      else:
        feature_name = 'Feature ' + str(i)

      plot_real_vs_predicted_feature(axs[i], real[:,i], predicted[:,i], predicted_offset, feature_name)

  else:

      fig, ax = plt.subplots(figsize = (16,8))

      if feature_names != None and len(feature_names) > 0:
        plot_real_vs_predicted_feature(ax, real, predicted, predicted_offset, feature_names[0])
      else:
        plot_real_vs_predicted_feature(ax, real, predicted, predicted_offset)
    
  plt.show()


# Converts the given list of folds to a single numpy array and returns the offset of each fold.
def unfold(y_folded):

  y = None
  fold_offset = 0
  fold_offsets = []

  for y_fold in y_folded:
    if y is None:
      y = y_fold
    else:
      y = np.vstack((y, y_fold))
      
    fold_offsets.append(fold_offset)
    fold_offset = fold_offset + y_fold.shape[0]

  return y, fold_offsets


def fold(y, K_folds):

    fld_idx = fold_indices(y.shape[0], K_folds)
    y_folds = np.vsplit(y, fld_idx) # list with folds

    return y_folds


def series_split(X, y, ratio=1.0):

  train_test_ratio = 0.75

  # Use y.shape[0] as last 'X' window doesn't have a 'y'

  if train_test_ratio < 1.0 and train_test_ratio > 0.0:

    train_size = int(y.shape[0]*train_test_ratio)

    if X.ndim > 2:
      X_test = X[train_size:y.shape[0],:,:]
    else:
      X_test = X[train_size:y.shape[0],:]

    if y.ndim > 1:
      y_test = y[train_size:y.shape[0],:]
    else:
      y_test = y[train_size:y.shape[0]]

  else:
    train_size = y.shape[0]
    X_test = None
    y_test = None

  if X.ndim > 2:
    X_train = X[:train_size,:,:]
  else:
    X_train = X[:train_size,:]
  
  if y.ndim > 1:
    y_train = y[:train_size,:]
  else:
    y_train = y[:train_size]

  return X_train, y_train, X_test, y_test


def read_ETD(spec='h1'):

  available_specs = ['h1','h2','m1','m2']

  if spec not in available_specs:
    spec = 'h1'

  etd_path = 'datasets/other/ETDataset/ETDataset-main/ETT-small/'
  dset_name = 'ETT' + spec
  etd_file = etd_path + dset_name + '.csv'
  ETT = pd.read_csv(etd_file, index_col=False, header=0, lineterminator='\n')
  ETT['date'] = pd.to_datetime(ETT['date'], format='%Y-%m-%d %H:%M:%S')

  ETT.set_index(ETT['date'], inplace=True)
  ETT.drop(['date'], axis=1, inplace=True)

  return ETT

def read_SML2010(project_root_path):

  sml_path = project_root_path + 'datasets/other/SML2010/NEW-DATA/'

  sml_file = sml_path + 'NEW-DATA-1.T15.txt'
  sml = pd.read_csv(sml_file, index_col=False, header=0, sep=' ', lineterminator='\n')

  dt = pd.to_datetime(sml['1:Date'] + ' ' +sml['2:Time'], format='%d/%m/%Y %H:%M')
  sml.set_index(dt, inplace=True)
  sml.drop(['1:Date','2:Time'], axis=1, inplace=True)

  all_zero_columns = ['19:Exterior_Entalpic_1','20:Exterior_Entalpic_2','21:Exterior_Entalpic_turbo']
  sml.drop(all_zero_columns, axis=1, inplace=True)

  no_zero_mask = (sml[sml.columns.difference(['12:Precipitacion'])] != 0.0) # '12:Precipitacion' column values are almost all zeros so it is excluded from the test
  sml = sml[no_zero_mask.all(axis=1)] # The dataset could contain missing values!

  return sml


def read_AirQuality(project_root_path):
  airqual_path = project_root_path + 'datasets/other/AirQualityDataSet/AirQualityUCI/'
  airqual_file = airqual_path + 'AirQualityUCI.csv'

  airqual = pd.read_csv(airqual_file, index_col=False, header=0, sep=';', decimal=',')

  dt = pd.to_datetime(airqual['Date'] + airqual['Time'], format='%d/%m/%Y%H.%M.%S')

  airqual.set_index(dt, inplace=True)
  airqual.drop(['Date','Time','Unnamed: 15','Unnamed: 16'], axis=1, inplace=True)

  airqual = airqual[(airqual != -200).all(axis=1)] # Missing values are tagged with -200 value.

  return airqual


def read_EnergyCo(project_root_path):
  energyco_path = project_root_path + 'datasets/other/Appliances_energy_prediction/'
  energyco_file = energyco_path + 'energydata_complete.csv'

  energyco = pd.read_csv(energyco_file, index_col=False, header=0, sep=',')

  dt = pd.to_datetime(energyco['date'], format='%Y-%m-%d %H:%M:%S')

  energyco.set_index(dt, inplace=True)
  energyco.drop(['date'], axis=1, inplace=True)

  return energyco


def read_poll(path):

    poll_file = path+'poll/presidential_polls_raw_national.csv'
    poll = pd.read_csv(poll_file, index_col=False, header=0, lineterminator='\n')

    clinton_col = 'rawpoll_clinton'
    trump_col = 'rawpoll_trump'

    poll.rename(columns={clinton_col:'Dem_poll', trump_col:'Rep_poll'}, inplace=True)
    poll['date'] = pd.to_datetime(poll['date'])
    poll = poll.set_index('date')
    poll = poll[poll.index>='2016-08-30']
    poll = poll.div(100.0)
    poll.name = 'poll' # Assign name here because copying a dataframe does not copy its name!

    return poll

def read_poll_all_parties(project_root_path=''):
  path = project_root_path+'datasets/'
  descriptors = read_desc_both_parties(path)
  polls = read_poll(path)
  desc_poll = align_concat(descriptors, polls)
  return desc_poll
  



def read_desc(path, party, filename, dfname):
  desc_file = path + 'twitter_descriptors/' + party + filename
  desc = pd.read_csv(desc_file, index_col=False, header=0, lineterminator='\n')
  desc.rename(columns={'Unnamed: 0':'date', 'offensive-1\r':'offensive-1'}, inplace=True)

  desc['date'] = pd.to_datetime(desc['date'])
  desc = desc.set_index('date')
  desc.name = dfname # Assign name here because copying a dataframe does not copy its name!
  #desc.plot(title='Descriptors', figsize=figure_size)
  desc.columns = [party + '_' + s for s in desc.columns]
  return desc

def read_desc_both_parties(path):

  dem_desc = read_desc(path, 'Dem', '_time_series_MEAN.csv', 'desc')
  rep_desc = read_desc(path, 'Rep', '_time_series_MEAN.csv', 'desc')
  desc = pd.concat([dem_desc, rep_desc], axis =1 )
  return desc


def align_concat(df1, df2):
    df1_aligned, df2_aligned = df1.align(df2, join='inner', axis=0)
    df1_df2 = pd.concat([df1_aligned, df2_aligned], axis =1 )
    return df1_df2

