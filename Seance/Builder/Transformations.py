# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss
from scipy import stats, special
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   PowerTransformer, MaxAbsScaler, QuantileTransformer)


class log:

    def __init__(self):
        pass

    def fit(self, y):
        pass

    def transform(self, y):
        transformed = np.log(y)
        return transformed

    def inverse_transform(self, y):
        transformed = np.exp(y)
        return transformed

class RobustBoxCox:
    def __init__(self):
        self.minmax = None
        self.boxcox = None

    def fit(self, y):
        self.minmax = MinMaxScaler(feature_range=(1, 2))
        self.minmax.fit(y.reshape(-1,1))
        trans_y = self.minmax.transform(y.reshape(-1,1))
        self.boxcox = PowerTransformer(method='box-cox')
        self.boxcox.fit(trans_y.reshape(-1,1))

    def transform(self, y):
        y_copy = y.copy()
        trans_y = self.minmax.transform(y_copy.reshape(-1,1))
        transformed = self.boxcox.transform(trans_y.reshape(-1,1))
        return transformed

    def inverse_transform(self, y):
        y_copy = y.copy().reshape(-1,1)
        inv_transf = self.minmax.inverse_transform(self.boxcox.inverse_transform(y_copy))
        if not np.isfinite(inv_transf.reshape(-1)).all():
            # y_copy = np.clip(y_copy, a_min=1, a_max=2)
            inv_transf = self.boxcox.inverse_transform(y_copy)
            y_mean = np.mean(y_copy)
            for idx, i in enumerate(zip(y_copy, inv_transf)):
                if np.isnan(i[1]):
                    if i[0] > y_mean:
                        inv_transf[idx] = 2
                    else:
                        inv_transf[idx] = 1
            inv_transf = self.minmax.inverse_transform(inv_transf)
        return inv_transf


class SeanceScaler:

    def __init__(self,
                 scaler,
                 scale,
                 linear_trend,
                 linear_test_window,
                 seasonal_period,
                 transformers):
        self.scaler = scaler
        self.scale = scale
        self.linear_trend = linear_trend
        self.linear_test_window = linear_test_window
        self.linear = False
        self.seasonal_period = seasonal_period
        self.predict = True
        factory_mapping = {'standard': StandardScaler(),
                           'minmax': MinMaxScaler(),
                           'maxabs': MaxAbsScaler(),
                           'robust': RobustScaler(),
                           'quantile': QuantileTransformer(),
                           'boxcox': PowerTransformer(method='box-cox'),
                           'log': log(),
                           'robust_boxcox': RobustBoxCox()
                            }
        self.transformer = factory_mapping[self.scaler]
        self.transformers = transformers


    def fit(self, y):
        if self.scale:
            self.transformer.fit(y)

    def get_deterministic_trend(self, y):
        trend_line, self.linear, slope, intercept, penalty = self.linear_test(y)
        if self.linear:
            series_level = np.mean(trend_line)
            trend_line = trend_line - series_level
            y = np.subtract(y.reshape((-1,)),
                            trend_line)
            self.trend_line = trend_line
            self.slope = slope
            self.intercept = trend_line[0]
            self.penalty = penalty
            self.series_level = series_level
        return y

    def linear_test(self, y):
        y = y
        xi = np.arange(1, len(y) + 1)
        # xi = xi**2
        slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y.reshape(-1, ))
        trend_line = slope*xi + intercept
        if self.linear_trend is True:
            linear = True
            return trend_line, linear, slope, intercept, r_value
        if self.seasonal_period is not None:
            required_len = 1.5 * self.seasonal_period
        else:
            required_len = 6
        if self.linear_trend == 'auto' and len(y) > required_len:
            if self.linear_test_window is not None:
                n_bins = self.linear_test_window
            else:
                n_bins = (1 + len(y)**(1/3) * 2)
            splitted_array = np.array_split(y.reshape(-1,), int(n_bins))
            mean_splits = np.array([np.mean(i) for i in splitted_array])
            asc_array = np.sort(mean_splits)
            desc_array = np.flip(asc_array)
            if all(asc_array == mean_splits):
                growth = True
            elif all(desc_array == mean_splits):
                growth = True
            else:
                growth = False
            if (r_value > .9 and growth):
                linear = True
            else:
                linear = False
        else:
            linear = False
        # slope = slope * r_value
        return trend_line, linear, slope, intercept, r_value

    def transform(self, y):
        y = y.values
        if self.linear_trend:
            y = self.get_deterministic_trend(y)
        if self.scale:
            self.fit(y.reshape(-1, 1))
            y = self.transformer.transform(y.reshape(-1, 1))
        self.transformers.append(self.inverse_transform)
        return np.array(y).reshape(-1)

    def retrend_predicted(self, y):
        slope = self.slope
        intercept = self.intercept
        penalty = self.penalty
        fit_trend = self.trend_line
        n = len(fit_trend)
        linear_trend = [i for i in range(0, len(y))]
        linear_trend = np.reshape(linear_trend, (len(linear_trend), 1))
        linear_trend += n + 1
        linear_trend = np.multiply(linear_trend, slope) + intercept
        y = np.add(y.reshape(-1), np.reshape(linear_trend, (-1,)))
        return y

    def retrend_fitted(self, y):
        trend = self.trend_line
        y = np.add(y.reshape(-1), trend)
        return y

    def inverse_transform(self, y, **kwargs):
        if self.scale:
            y = self.transformer.inverse_transform(y.reshape((-1,1)))
        if self.linear:
            if self.predict is None:
                y = self.retrend_fitted(y)
            else:
                y = self.retrend_predicted(y)
        if self.predict is None:
            self.predict = True
        return y