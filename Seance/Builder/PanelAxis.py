# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from Seance.basis_functions.linear_basis import LinearBasisFunction

class PanelAxis:
    
    def __init__(self,
                 run_dict,
                 n_basis,
                 decay,
                 weighted,
                 seasonal_period,
                 basis_difference):
        self.run_dict = run_dict
        self.seasonal_period = seasonal_period
        if not isinstance(n_basis, list) and n_basis is not None:
            n_basis = [n_basis]
        self.n_basis = n_basis
        self.decay = decay
        self.basis_difference = basis_difference
        self.weighted = weighted
    
    def get_piecewise(self, y, n_basis, ts_id):
        if n_basis >= len(y):
            n_basis = max(1, len(y) - 1)
        lbf = LinearBasisFunction(n_changepoints=n_basis,
                                  decay=self.decay,
                                  weighted=self.weighted,
                                  basis_difference=self.basis_difference)
        basis = lbf.get_basis(y)
        self.run_dict['local'][ts_id][f'{n_basis}_function'] = lbf
        self.run_dict['local'][ts_id][f'{n_basis}_basis'] = basis
        return basis

    def build_axis(self, dataset):
        ts_id = dataset['Murmur ID'].iloc[0]
        if self.n_basis is not None and self.n_basis:
            for basis in self.n_basis:
                linear_basis = self.get_piecewise(dataset['Murmur Target'],
                                                  basis,
                                                  ts_id)
                size = np.shape(linear_basis)[1] - 1
                linear_basis = pd.DataFrame(linear_basis,
                                            index=dataset.index,
                                            columns=[f'{basis}_basis_{i}' for i in range(size)]+ ['Trend'])
                if 'Trend' in dataset.columns:
                    linear_basis = linear_basis.drop('Trend', axis=1)
                dataset = pd.concat([dataset, linear_basis],
                                    axis=1)
        return dataset
    
    def build_future_axis(self, refined_df, forecast_horizon, ts_id):
        id_dict = self.run_dict['local'][ts_id]
        final_basis = []
        if self.n_basis is not None and self.n_basis:
            for basis in self.n_basis:
                try:
                    X = id_dict[f'{basis}_function'].get_future_basis(id_dict[f'{basis}_basis'],
                                                              forecast_horizon)
                except:
                    X = np.resize(np.array(np.nan), (forecast_horizon, basis))
                size = np.shape(X)[1] - 1
                X = pd.DataFrame(X,
                                 index=refined_df.index,
                                 columns=[f'{basis}_basis_{i}' for i in range(size)] + ['Trend'])
                if final_basis:
                    X = X.drop('Trend', axis=1)
                else:
                    X['Murmur ID'] = ts_id
                    X[self.run_dict['global']['Date Column']] = refined_df[self.run_dict['global']['Date Column']].values
                final_basis.append(X)
            return pd.concat(final_basis, axis=1)
        else:
            return None