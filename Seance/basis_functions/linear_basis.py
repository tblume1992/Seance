from numba import njit
import numpy as np


@njit
def get_basis(y, n_changepoints, decay):
    y = y.copy()
    y -= y[0]
    n = len(y)
    mean_y = np.mean(y)
    changepoints = np.zeros(shape=(len(y), n_changepoints))
    if n_changepoints > n - 1:
        n_changepoints = n - 1
    array_splits = []
    n_trunc = len(y)

    for i in range(1, n_changepoints + 1):
        i = n_changepoints - i + 1
        split_point = n_trunc//i
        array_splits.append(y[:split_point])
        n_trunc -= len(array_splits[-1])
        y = y[split_point:]
    len_splits = 0
    initial_point = np.mean(array_splits[0])
    final_point = np.mean(array_splits[-1])
    for i in range(n_changepoints):
        len_splits += len(array_splits[i])
        moving_point = np.mean(array_splits[i])
        left_basis = np.linspace(initial_point,
                                  moving_point,
                                  len_splits)
        if decay is None:
            end_point = final_point
        else:
            if decay == -1:
                if not mean_y:
                    mean_y += 0.00001
                dd = moving_point**2 / (mean_y**2)
                if dd > .99:
                    dd = .99
                if dd < .001:
                    dd = .001
                end_point = moving_point - ((moving_point - final_point) * (1.0 - dd))
            else:
                end_point = moving_point - ((moving_point - final_point) * (1.0 - decay))
        right_basis = np.linspace(moving_point,
                                  end_point,
                                  n - len_splits + 1)
        changepoints[:, i] = np.append(left_basis, right_basis[1:])
    slopes = changepoints[-1] - changepoints[-2]
    diff = changepoints[-1, :]
    return changepoints, slopes, diff

@njit
def get_future_basis(forecast_horizon, slopes, diff, n_changepoints, training_length):
    future_basis = np.arange(0, forecast_horizon + 1)
    future_basis += training_length
    future_basis = future_basis.reshape(-1, 1)
    for i in np.arange(1, n_changepoints):
        future_basis = np.append(future_basis, future_basis[:, :1], axis=1)
    future_basis = future_basis * slopes
    future_basis = future_basis + (diff - future_basis[0, :])#diff
    return future_basis[1:, :]


#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seasonality = ((np.cos(np.arange(1, 101))*10 + 50))
    np.random.seed(100)
    true = np.linspace(-1, 1, 100)
    noise = np.random.normal(0, 1, 100)
    y = true + noise + seasonality
    plt.plot(y)
    basis, slopes, diff = get_basis(y, 10, decay=-1)
    future_basis = get_future_basis(24, slopes, diff, 10, len(y))
    full = np.append(basis, future_basis, axis=0)
    plt.plot(full + y[0])