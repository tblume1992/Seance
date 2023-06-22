# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import shap
sns.set_style('darkgrid')


def rename_column(column_name):
    if 'basis' in column_name:
        return 'Trend'
    if 'fourier' in column_name:
        return 'Seasonality'
    if 'ar_' in column_name:
        return 'AutoRegressiveLag'
    return column_name

class Explainer:

    def __init__(self, seance_obj, id_column, date_column):
        self.seance_obj = seance_obj
        self.model_obj = seance_obj.model_obj
        self.id_column = id_column
        self.date_column = date_column

    def explain(self, train_X, pred_X):
        pred_X = pred_X[train_X.columns]
        explainer = shap.TreeExplainer(self.model_obj)
        shap_values = explainer.shap_values(pred_X)
        shap_values = pd.DataFrame(shap_values, columns=pred_X.columns)
        shap_values = shap_values.rename({'Seance ID': 'Level'}, axis=1)
        shap_values[self.id_column] = pred_X[self.id_column].values
        shap_values['period'] = pred_X['period'].values
        shap_values['Seance ID'] = pred_X['Seance ID'].values
        shap_values[self.date_column] = shap_values[self.date_column]
        return shap_values

    def plot(self, seance_ids, shap_values):
        if not isinstance(seance_ids, list):
            seance_ids = [seance_ids]
        for forecast in seance_ids:
            refined = shap_values[shap_values['Seance ID'] == forecast]
            refined.columns = [rename_column(i) for i in refined.columns]
            refined = refined.groupby(level=0, axis=1).sum()
            refined = refined.set_index(self.date_column)
            refined = refined.drop([self.id_column, 'Seance ID'],axis=1)
        # def re_transform(values):
        #     print(values.values)
        #     values = np.array(values.values)
        #     return run_dict['local'][forecast]['scaler'].inverse_transform(values)
        # refined = refined.apply(re_transform,
        #                         axis=0,
        #                         )
            refined.plot(kind='bar', stacked=True)

#%%
if __name__ == '__main__':
    exp = Explainer(seance_obj=seance, id_column='ID', date_column='Datetime')
    exp.explain(seance.processed_df, seance.pred_X)
    model_obj = seance.mlforecast.models_['LGBMRegressor']
