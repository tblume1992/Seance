# -*- coding: utf-8 -*-

from numba import njit, jit
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Seance.Builder.Transformations import SeanceScaler

def cap_outliers(values, outlier_cap):
    """
    

    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    outlier_cap : TYPE
        DESCRIPTION.

    Returns
    -------
    series : TYPE
        DESCRIPTION.

    """
    mean = np.mean(values)
    std = np.std(values)
    values = np.clip(values,
                     a_min=mean - outlier_cap * std,
                     a_max=mean + outlier_cap * std)
    return values



class PreProcess:
    def __init__(self,
                 scale,
                 scale_type,
                 id_column,
                 target_column,
                 linear_trend,
                 linear_test_window,
                 seasonal_period,
                 outlier_cap,
                 run_dict):
        self.scale = scale
        self.scale_type = scale_type
        self.id_column = id_column
        self.target_column = target_column
        self.linear_trend = linear_trend
        self.seasonal_period = seasonal_period
        self.linear_test_window = linear_test_window
        self.seasonal_period = seasonal_period
        self.outlier_cap = outlier_cap
        self.run_dict = run_dict
        self._transformers = []

    def create_seance_id(self, dataset):
        try:
            dataset[self.id_column] = dataset[self.id_column].astype(int)
            dataset['Seance ID'] = dataset[self.id_column]
            self.run_dict['global']['ID Mapping'] = dataset[[self.id_column, 'Seance ID']].drop_duplicates()
        except:
            le = LabelEncoder()
            dataset['Seance ID'] = le.fit_transform(dataset[self.id_column].values)
        self.run_dict['global']['ID Mapping'] = dataset[[self.id_column, 'Seance ID']].drop_duplicates()
        return dataset

    def trans(self, df):
        _scaler = SeanceScaler(scaler=self.scale_type,
                               scale=self.scale,
                               linear_trend=self.linear_trend,
                               linear_test_window=self.linear_test_window,
                               seasonal_period=self.seasonal_period,
                               transformers=self._transformers)
        df[self.target_column] = _scaler.transform(df[self.target_column])
        return df

    def process(self, df):
        df = self.create_seance_id(df)
        if self.scale:
            df = df.groupby('Seance ID').apply(self.trans)
        if self.outlier_cap is not None:
            df[self.target_column] = df.groupby('Seance ID')[self.target_column]\
                                        .transform(cap_outliers,
                                        outlier_cap=self.outlier_cap)
        return df



#%%

if __name__=='__main__':
    from mlforecast.utils import generate_daily_series
    from window_ops.expanding import expanding_mean
    from window_ops.rolling import rolling_mean
    from mlforecast.target_transforms import Differences
    series = generate_daily_series(
        n_series=20,
        max_length=100,
        n_static_features=1,
        static_as_categorical=False,
        with_trend=True
    )
    run_dict = {'global': {}}
    processor = PreProcess(scale=True,
                           scale_type='standard',
                           id_column='unique_id',
                           target_column='y',
                           linear_trend='auto',
                           linear_test_window=None,
                           seasonal_period=None,
                           outlier_cap=5,
                           run_dict=run_dict)
    processed_df = processor.process(series)
    look = processor._transformers
