# -*- coding: utf-8 -*-

from numba import njit
import numpy as np


@njit
def get_fourier_series(length, seasonal_period, fourier_order):
    x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
    t = np.arange(1, length + 1).reshape(-1, 1)
    x = x * t
    fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return fourier_series

@njit
def get_future_fourier(length, forecast_horizon, seasonal_period, fourier_order):
    x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
    t = np.arange(1, length + 1 + forecast_horizon).reshape(-1, 1)
    x = x * t
    fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return fourier_series[-forecast_horizon:]

#%%
if __name__ == '__main__':
    f = get_fourier_series(300, 12, 10)
    ff = get_future_fourier(len(f), 50, 12, 10)
