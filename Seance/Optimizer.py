# -*- coding: utf-8 -*-
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import optuna
from numba import njit
import numpy as np
from Seance.Forecaster import Forecaster
# optuna.logging.set_verbosity(optuna.logging.WARNING)


@njit
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (0.0001 + (np.abs(A) + np.abs(F))))
def grouped_smape(df, target_column):
    return smape(df[target_column].values, df['LGBMRegressor'].values)

class Optimize:
    def __init__(self,
                 df,
                 target_column,
                 date_column,
                 id_column,
                 freq,
                 test_size,
                 metric='mse',
                 seasonal_period=0,
                 n_folds=1, #TODO
                 n_trials=100,
                 max_n_estimators=500,
                 ar_lags=None,
                 timeout=None):
        self.df = df.sort_values([id_column, date_column])
        if isinstance(seasonal_period, list):
            self.max_pulse = max(seasonal_period)
        else:
            self.max_pulse = seasonal_period
        self.seasonal_period = seasonal_period
        self.n_folds = n_folds
        self.test_size = test_size
        self.n_trials = n_trials
        self.target_column = target_column
        self.date_column = date_column
        self.id_column = id_column
        self.timeout = timeout
        self.freq = freq
        self.max_n_estimators = max_n_estimators
        # if ar_lags is None and seasonal_period is not None:
        #     ar_lags = list(np.arange(1, self.max_pulse + 1))
        # if ar_lags is None and seasonal_period is None:
        #     ar_lags = list(np.arange(1, 13))
        self.ar_lags = ar_lags
        self.metric= metric

    # def logic_layer(self):
    #     n_samples = len(y)
    #     test_size = n_samples//(self.n_folds + 1)
    #     if n_samples - test_size < self.max_pulse:
    #         self.seasonal_period = 0

    def get_splits(self, df, id_column):
        df['test_split'] = df.groupby(id_column).cumcount()+1
        df['len'] = df.groupby(id_column)[self.date_column].transform('size')
        df['test_split'] = ((df['test_split'] - self.test_size > self.test_size) & (df['test_split'] > df['len'] - self.test_size))
        self.train_df = df[df['test_split'] == False]
        self.test_df = df[df['test_split'] == True]

    def scorer(self, params, metric):
        scores = []
        # for train_index, test_index in cv_splits:
        try:
            model_obj = Forecaster()
            model_obj.fit(self.train_df,
                          target_column=self.target_column,
                          date_column=self.date_column,
                          id_column=self.id_column,
                          freq=self.freq,
                          **params)
            predicted = model_obj.predict(self.test_size)
            self.predicted = predicted
            assert len(predicted) == len(self.test_df), 'Predicted not the same size as test set'
    
            if any(np.isnan(predicted['LGBMRegressor'])):
                scores.append(np.inf)
            else:
                self.cv_df = self.test_df[[self.id_column, self.date_column, self.target_column]].merge(predicted, on=[self.id_column, self.date_column])
                if metric == 'mse':
                    scores.append(mean_squared_error(self.cv_df[self.target_column].values, self.cv_df['LGBMRegressor'].values))
                elif metric == 'smape':
                    scores.append(self.cv_df.groupby(self.id_column).apply(grouped_smape, self.target_column))
        except:
            scores.append(np.inf)
        return np.mean(scores)


    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int(name="n_estimators", low=50, high=self.max_n_estimators),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            "use_id": trial.suggest_categorical("use_id", [True, False]),
            "objective": trial.suggest_categorical("objective", ['regression', 'regression_l1']),
            "n_basis": trial.suggest_int("n_basis", 0, 25),
            "differences": trial.suggest_categorical("differences", [None, 1]),
            "decay": trial.suggest_categorical("decay", [-1,
                                                         .05,
                                                         .1,
                                                         .25,
                                                         .5,
                                                         .75,
                                                         .9,
                                                         .99]),
            "scale_type": trial.suggest_categorical("scale_type", ['log',
                                                                   'standard',
                                                                    'minmax',
                                                                   # 'maxabs',
                                                                   # 'robust',
                                                                   # 'boxcox',
                                                                   'robust_boxcox',
                                                                   'none']),
        }
        if self.seasonal_period:
            params.update({'seasonal_period': trial.suggest_categorical("seasonal_period", [None, self.seasonal_period])})
        if self.ar_lags is not None:
            params.update({'lags': trial.suggest_categorical("lags", self.ar_lags)})
        else:
            if self.seasonal_period:
                params.update({'lags': trial.suggest_int(name="lags", low=1, high=self.max_pulse + 1)})
                params['lags'] = list(range(1, params['lags']))
            else:
                params.update({'lags': trial.suggest_int(name="lags", low=1, high=13)})
                params['lags'] = list(range(1, params['lags']))
        score = self.scorer(params, self.metric)
        return score

    def fit(self, seed):
        self.get_splits(self.df, self.id_column)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(self.objective, n_trials=self.n_trials)
        best_params = study.best_params
        if best_params['lags'] == 1:
            best_params.update({'lags': [study.best_params['lags']]})
        else:
            best_params.update({'lags': list(range(1, best_params['lags']))})
        return best_params, study

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    opt = Optimize(train_df[['V', 'Datetime', 'ID']],
                target_column='V',
                date_column='Datetime',
                id_column='ID',
                freq='W',
                seasonal_period=52,
                test_size=26,
                ar_lags=[list(range(1, 53))],
                n_trials=3)
    study = opt.fit(seed=1)

    look = opt.test_df
    look2 = opt.predicted
    merged = opt.test_df.merge(opt.predicted, on=['ID', 'Datetime'])
    optuna.visualization.matplotlib.plot_param_importances(study)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    optuna.visualization.plot_contour(study).show(renderer="browser")
    best_params = study.best_params
    # best_params['lags'] = list(range(1, 53))


    study.best_value
    look = opt.cv_df.groupby('ID').apply(grouped_smape, 'V')
    look1 = opt.train_df
    look2 = opt.test_df

    train_df['test_split'] = train_df.groupby('ID').cumcount()+1
    train_df['len'] = train_df.groupby('ID')['Datetime'].transform('size')
    train_df['split'] = ((train_df['test_split'] - 26 >= 26) & (train_df['test_split'] >= train_df['len'] - 26))


    # seance = Forecaster(floor=0)
    # output = seance.fit(train_df[['V', 'Datetime', 'ID']],
    #             num_leaves=100,
    #             learning_rate=0.1,
    #             n_estimators=31,
    #             # n_basis=3,
    #             target_column='V',
    #             date_column='Datetime',
    #             id_column='ID',
    #             freq=freq,
    #             use_id=False,
    #             scale=False,
    #             seasonal_period=None,
    #             scale_type='log',
    #             differences=1,
    #             # lags=[24 * i for i in range(1, 15)],
    #             lags=list(range(1, 13)),
    #             num_threads=4,
    #             lag_transforms={
    #     # 24: [(ewm_mean, 0.3), (rolling_mean, 7 * 24), (rolling_mean, 7 * 48)],
    #     # 48: [(ewm_mean, 0.3), (rolling_mean, 7 * 24), (rolling_mean, 7 * 48)],
    # }
    # )