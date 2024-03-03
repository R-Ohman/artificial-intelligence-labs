import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data


data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

x_train = 1/x_train
x_test = 1/x_test

test_size = x_test.size


def get_observation_matrix(x):
    # 1.8: observation matrix = [[1], [1], ... [1]] + x
    x = x.reshape(-1, 1)
    ones = np.ones((x.size, 1))
    obs = np.concatenate((ones, x), axis=1)
    return obs


# TODO: calculate closed-form solution
def get_theta_best(x, y):
    obs = get_observation_matrix(x)
    # o - observations
    # 1.13: (o.T * o)^-1 * o.T * y
    inv_product = np.linalg.inv(np.matmul(obs.T, obs))
    ind_vars = np.matmul(np.matmul(inv_product, obs.T), y)
    return ind_vars


theta_best = get_theta_best(x_train, y_train)
print("Theta = ", theta_best)


# TODO: calculate error
def get_mse(x_arr, y_arr):
    mse = (1 / test_size) * sum((theta_best[0] + theta_best[1] * x - y) ** 2 for x, y in zip(x_arr, y_arr))
    return mse


print("MSE = ", get_mse(x_test, y_test))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# 1.15: z = (x - mean) / std
x_mean = np.mean(x_train)
x_std = np.std(x_train)

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# TODO: calculate theta using Batch Gradient Descent
theta_best = np.random.rand(1, 2).reshape(-1, 1)  # random initialization of 2 x 1 matrix
learning_rate = 0.01
x_obs = get_observation_matrix(x_train)

for i in range(0, 100000):
    gradientMse = (2 / test_size) * x_obs.T.dot(x_obs.dot(theta_best) - y_train.reshape(-1, 1))
    theta_best = theta_best - learning_rate * gradientMse


theta_best = theta_best.flatten()
print("Theta = ", theta_best)

# TODO: calculate error
print("MSE = ", get_mse(x_test, y_test))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x

# destandardization
x = x * x_std + x_mean
x_test = x_test * x_std + x_mean
x = 1/x
x_test = 1/x_test

plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
