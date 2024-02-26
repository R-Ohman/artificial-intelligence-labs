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

# TODO: calculate closed-form solution
def get_theta_best(x, y):
    # 1.8: observation matrix = [1 ; 1 ; ... 1] + x
    x = x.reshape(-1, 1)
    ones = np.ones((len(x), 1))
    obs = np.concatenate((ones, x), axis=1)
    # o - observations
    # 1.13: (o.T * o)^-1 * o.T * y
    ind_vars = np.linalg.inv(obs.T.dot(obs)).dot(obs.T).dot(y)
    return ind_vars


theta_best = get_theta_best(x_train, y_train)

# TODO: calculate error
def MSE(x, y, theta):
    # MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2
    error_sum = 0
    for i in range(len(x)):
        error_sum += ((theta[0] + theta[1] * x[i]) - y[i]) ** 2
    return error_sum

print("Error: ", MSE(x_test, y_test, theta_best))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# z = (x-m)/q
# x to pierwotna wartość zmiennej, µ to średnia z populacji, a σ to
# odchylenie standardowe populacji


# TODO: calculate theta using Batch Gradient Descent

# TODO: calculate error

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()