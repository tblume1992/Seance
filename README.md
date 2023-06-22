#Seánce v0.0.1

A simple wrapper around Nixtla's MLForecast aimed at streamlining plug-and-play forecasting.

A general pattern is to optimize then forecast such as:

```
from mlforecast.utils import generate_daily_series
series = generate_daily_series(
    n_series=20,
    max_length=100,
    min_length=50,
    with_trend=True
)


from Seance.Optimizer import Optimize
opt = Optimize(series,
            target_column='y',
            date_column='ds',
            id_column='unique_id',
            freq='D',
            seasonal_period=7,
            test_size=10,
            # ar_lags=[list(range(1, 8))], #by default this will be done based on seasonal period
            metric='smape',
            n_trials=100)
#returns an optuna study obj
best_params, study = opt.fit(seed=1)
```
#optuna plotting
```
import optuna
optuna.visualization.matplotlib.plot_param_importances(study)
```
![alt text](https://github.com/tblume1992/Seance/blob/main/static/seance_param_imp.png?raw=true "Param Importance")
Here we can see the most important parameter is (unsurprisingly) the number of lags. Followd by decay which controls the 'forgetfulness' of the basis functions.
```
optuna.visualization.matplotlib.plot_optimization_history(study)
```
![alt text](https://github.com/tblume1992/Seance/blob/main/static/seance_study.png?raw=true "Study")
#passing off best params for forecasts
```
seance = Forecaster()
output = seance.fit(series,
                    target_column='y',
                    date_column='ds',
                    id_column='unique_id',
                    freq='D',
                    **best_params)
predicted = seance.predict(24)
```
#quick plot of the forecasts
```
import matplotlib.pyplot as plt
plot_ser = np.append(series[series['unique_id'] == 'id_00']['y'].values,
                     predicted[predicted['unique_id'] == 'id_00']['LGBMRegressor'].values)
plt.plot(plot_ser)
plt.vlines(x=len(plot_ser) - 24, ymin=0, ymax=max(plot_ser), linestyle='dashed', color='red')
plt.show()
```
![alt text](https://github.com/tblume1992/Seance/blob/main/static/seance_forecast.png?raw=true "Forecast Results")