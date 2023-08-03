# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
from mlforecast import MLForecast
from Seance.Builder import PreProcess
from Seance.basis_functions.fourier_basis import get_fourier_series, get_future_fourier
from Seance.basis_functions.linear_basis import get_basis, get_future_basis

class Forecaster:
    def __init__(self,
                 floor=None,
                 ):
        self.floor = floor
        self.run_dict = {'global': {},
                         'local': {}}
        self.builder = None

    def linear_basis(self, y):
        basis, slopes, diff = get_basis(y.values, n_changepoints=self.n_basis, decay=self.decay)
        self.basis_list.append((slopes, diff, len(y)))
        return pd.DataFrame(basis,
                            columns=[f'{self.n_basis}_basis_{i}' for i in range(self.n_basis)],
                            index=y.index)

    def future_linear_basis(self, df):
        slopes, diff, training_length = self.basis_list[df['Seance ID'].iloc[0]]
        basis = get_future_basis(self.forecast_horizon, slopes, diff, self.n_basis, training_length)
        return pd.DataFrame(basis,
                            columns=[f'{self.n_basis}_basis_{i}' for i in range(self.n_basis)],
                            index=df.index)


    def fit(self,
            df,
            target_column,
            id_column,
            date_column,
            freq,
            model='lightgbm',
            alpha=None,
            time_exogenous=None,
            id_exogenous=None,
            id_feature_columns=None,
            time_feature_columns=None,
            scale=True,
            scale_type='standard',
            differences=None,
            categorical_columns=None,
            decay=-1,
            lags=None,
            use_id=True,
            ma=None,
            fourier_order=10,
            seasonal_weights=None,
            weighted=True,
            n_basis=None,
            seasonal_period=None,
            test_size=None,
            bagging_freq=1,
            linear_trend='auto',
            objective='regression',
            metric='rmse',
            learning_rate=.1,
            min_child_samples=5,
            num_leaves=50,
            n_estimators=50,
            return_proba=False,
            boosting_params=None,
            early_stopping_rounds=10,
            is_unbalance=True,
            scale_pos_weights=None,
            labels=None,
            basis_difference=False,
            linear_test_window=None,
            seasonal_dummy=False,
            outlier_cap=None,
            ts_features=False,
            sample_weights=None,
            num_threads=1,
            verbose=-1,
            lambda_l1=0.0,
            lambda_l2=0.0,
            bagging_fraction=1.0,
            feature_fraction=1.0,
            **kwargs,
            ):
        if scale_type == 'none':
            scale=False
        self.scale = scale
        self.model = model
        self.seasonal_period = seasonal_period
        self.fourier_order = fourier_order
        self.date_column = date_column
        self.id_column = id_column
        self.freq = freq
        self.n_basis = n_basis
        self.decay = decay
        forecast_columns = ['Seance ID',
                            date_column,
                            target_column]
        if categorical_columns is None:
            if use_id:
                categorical_columns = ['cat_id_col']
        else:
            if use_id:
                categorical_columns += ['cat_id_col']
        df = df.copy().sort_values([id_column, date_column])
        self.time_periods = df[date_column].drop_duplicates().sort_values()
        seasonal_df = []
        if self.seasonal_period is not None:
            if not isinstance(self.seasonal_period, list):
                self.seasonal_period = [self.seasonal_period]
            for i in self.seasonal_period:
                fourier_basis = get_fourier_series(length=len(self.time_periods),
                                                   seasonal_period=i,
                                                   fourier_order=self.fourier_order)
                column_names = [f'{i}_fourier_{j+1}' for j in range(2 * self.fourier_order)]
                fourier_basis = pd.DataFrame(fourier_basis, columns=column_names)
                seasonal_df.append(fourier_basis)
            seasonal_df = pd.concat(seasonal_df)
            seasonal_df[date_column] = self.time_periods.values
            seasonal_df[date_column] = pd.to_datetime(seasonal_df[date_column]).dt.tz_localize(None)
            max_period = max(self.seasonal_period)
        else:
            max_period = None

        if differences is not None:
            if not isinstance(differences, list):
                differences = [differences]
        self.processor = PreProcess.PreProcess(scale=scale,
                                          scale_type=scale_type,
                                          id_column=id_column,
                                          target_column=target_column,
                                          linear_trend=linear_trend,
                                          linear_test_window=linear_test_window,
                                          seasonal_period=max_period,
                                          outlier_cap=outlier_cap,
                                          run_dict=self.run_dict)
        self.processed_df = self.processor.process(df)
        self.processed_df = self.processed_df.sort_values(['Seance ID', date_column])
        self.processed_df[date_column] = self.processed_df[date_column].dt.tz_localize(None)
        self.last_dates = self.processed_df[['Seance ID', date_column]].sort_values([date_column]).groupby('Seance ID').tail(1)
        self.last_dates = self.last_dates.sort_values('Seance ID')
        if self.n_basis is not None and self.n_basis:
            self.basis_list = []
            basis = self.processed_df.groupby('Seance ID')[target_column].apply(self.linear_basis)
            self.processed_df = pd.concat([self.processed_df, basis], axis=1)
        if self.seasonal_period is not None:
            self.processed_df = self.processed_df.merge(seasonal_df, on=date_column)
        self.processed_df['cat_id_col'] = self.processed_df['Seance ID'].copy()
        if categorical_columns is not None:
            forecast_columns += categorical_columns
            for column in categorical_columns:
                self.processed_df[column] = self.processed_df[column].astype('category')
        if self.seasonal_period is not None:
            forecast_columns += [i for i in list(seasonal_df.columns) if i != date_column]
        if self.n_basis is not None and self.n_basis:
            forecast_columns += list(basis.columns)
        if self.model == 'lightgbm':
            lgb_params = {'sample_weights': sample_weights,
                          'metric': metric,
                          'learning_rate':learning_rate,
                          'min_child_samples':min_child_samples,
                          'num_leaves':num_leaves,
                          'n_estimators':n_estimators,
                          'learning_rate':learning_rate,
                          'num_threads': num_threads,
                           'bagging_freq': bagging_freq,
                          'verbose': verbose,
                            'lambda_l1': lambda_l1,
                            'lambda_l2': lambda_l2,
                            'bagging_fraction': bagging_fraction,
                            'feature_fraction': feature_fraction,
                            'objective': objective,
                            }
            self.mlforecast = MLForecast(models=[lgb.LGBMRegressor(**lgb_params)],
                                         freq=freq,
                                         lags=lags,
                                         differences=differences,
                                         **kwargs)
            self.pred_col = 'LGBMRegressor'
        elif self.model == 'linear_regression':
            self.mlforecast = MLForecast(models=[LinearRegression()],
                                         freq=freq,
                                         lags=lags,
                                         differences=differences,
                                         **kwargs)
            self.pred_col = 'LinearRegression'
        elif self.model == 'ridge':
            if alpha is None:
                alpha = 1
            self.mlforecast = MLForecast(models=[Ridge(alpha=alpha)],
                                         freq=freq,
                                         lags=lags,
                                         differences=differences,
                                         **kwargs)
            self.pred_col = 'Ridge'
        fitted = self.mlforecast.fit(self.processed_df[forecast_columns],
                                    id_col='Seance ID',
                                    time_col=date_column,
                                    target_col=target_column,
                                    static_features=categorical_columns)
        self.model_obj = fitted.models_[self.pred_col]
        return fitted

    def inverse_transform(self, df):
        seance_id = df['Seance ID'].iloc[0]
        unscaled = self.processor._transformers[seance_id](df[self.pred_col].values)
        unscaled = np.array(unscaled).reshape(-1)
        df[self.pred_col] = unscaled
        return df

    def predict(self, forecast_horizon):
        self.forecast_horizon = forecast_horizon
        future_dates = pd.date_range(start=self.time_periods.iloc[-1],
                                     freq=self.freq,
                                     periods=forecast_horizon + 1)[1:]
        full_dates = pd.concat([self.time_periods, pd.Series(future_dates)])
        id_df = self.run_dict['global']['ID Mapping']
        uids = pd.Series(np.repeat(id_df['Seance ID'].values, forecast_horizon),
                         name='Seance ID',
                         dtype='category')
        pred_dates = []
        for date in self.last_dates[self.date_column]:
            pred_dates += list(pd.date_range(start=date,
                              freq=self.freq,
                              periods=forecast_horizon + 1)[1:])
        pred_X = pd.DataFrame(uids, columns=['Seance ID'])
        pred_X[self.date_column] = pred_dates
        pred_X[self.date_column] = pred_X[self.date_column].dt.tz_localize(None)
        if self.seasonal_period is not None:
            seasonal_df = []
            for i in self.seasonal_period:
                fourier_basis = get_future_fourier(forecast_horizon=forecast_horizon,
                                                   length=len(self.time_periods),
                                                   seasonal_period=i,
                                                   fourier_order=self.fourier_order)
                column_names = [f'{i}_fourier_{j+1}' for j in range(2 * self.fourier_order)]
                fourier_basis = pd.DataFrame(fourier_basis, columns=column_names).astype(float)
                seasonal_df.append(fourier_basis)
            seasonal_df = pd.concat(seasonal_df)
            seasonal_df[self.date_column] = full_dates.values
            pred_X = pred_X.merge(seasonal_df, on=self.date_column)
        if self.n_basis is not None and self.n_basis:
            basis = pred_X.groupby('Seance ID').apply(self.future_linear_basis)
            pred_X = pd.concat([pred_X, basis], axis=1)
        if self.seasonal_period is not None or (self.n_basis is not None and self.n_basis):
            self.dynamic_dfs = [pred_X]
        else:
            self.dynamic_dfs = None
        self.pred_X = pred_X
        predicted = self.mlforecast.predict(horizon=forecast_horizon,
                                            dynamic_dfs=self.dynamic_dfs)
        predicted = predicted.merge(self.run_dict['global']['ID Mapping'],
                                    on='Seance ID')
        if self.scale:
            predicted = predicted.groupby('Seance ID').apply(self.inverse_transform)
        predicted[self.pred_col] = predicted[self.pred_col].clip(lower=self.floor)
        predicted = predicted.sort_values([self.id_column, self.date_column])
        if any(np.isnan(predicted[self.pred_col])):
            print('Nan found when Inverse Transforming, use a more stable transformer such as "standard"')
        return predicted

    def plot_importance(self, max_num_features=20):
        lgb.plot_importance(self.mlforecast.models_[self.pred_col],
                            max_num_features=max_num_features)
        return


