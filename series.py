import numpy as np

def descale(arr, scaler):
    scaler.fit(arr)    
    return scaler.transform(arr)

def detrend(arr):
    arr = np.diff(arr, axis = 0)
    return arr

def retrend(arr, first_row):

    retrended = np.vstack((first_row, arr))

    for i in range(1, retrended.shape[0]):
        retrended[i] += retrended[i-1]

    return retrended


def rescale(arr, scaler):
    return scaler.inverse_transform(arr)


def normalize(arr, scaler):
    # arr = detrend(arr)
    arr = descale(arr, scaler)
    return arr

def denormalize(arr, scaler, bias=None):
    arr = rescale(arr, scaler)
    """
    if bias is not None:
        arr = retrend(arr, bias)
    """
    return arr

def denormalize_fold_dict(fold_dict, scaler, bias=None):
    for offset in fold_dict:
        fold_dict[offset] = denormalize(fold_dict[offset], scaler, bias)


import matplotlib.pyplot as plt

def plot_series(ax, series):
  t = np.arange(series.shape[0])
  ax.plot(t, series, label='Real')


def plot_np_array(arr):

    n_series = arr.shape[-1]

    fig, axs = plt.subplots(1, n_series, figsize = (16,8))

    for i in range(n_series):
        plot_series(axs[i], arr[:,i])

    plt.show()