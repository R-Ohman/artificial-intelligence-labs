import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data


def create_x_matrix(data):
    weight = 1 / data['Weight'].to_numpy()
    horsepower = 1 / data['Horsepower'].to_numpy()
    displacement = 1 / data['Displacement'].to_numpy()
    model_year = data['Model Year'].to_numpy() ** 2
    ones = np.ones_like(weight)
    return np.c_[ones, weight]


def calculate_optimal_theta(x, y):
    # 1.13: (x.T * x)^-1 * x.T * y
    inv_product = np.linalg.inv(np.matmul(x.T, x))
    return np.matmul(np.matmul(inv_product, x.T), y)


def calculate_mse(x, y, theta):
    """
    size = x.shape[0]
    mse = 0
    for i in range(size):
        # 1.2: y_prediction = theta_0 + theta_1 * x + ... + theta_n * x
        prediction = np.dot(x[i], theta)
        mse += (y[i] - prediction) ** 2
    mse /= size
    """
    return sum([((y[i] - np.dot(x[i], theta)) ** 2) for i in range(x.shape[0])]) / x.shape[0]


def calculate_gradient_descent_optimal_theta(x):
    # start with random theta
    theta = np.random.rand(1, params_number).reshape(-1, 1)
    learning_rate = 0.01

    train_size = x.shape[0]
    for _ in range(10000):
        gradient = (2 / train_size) * (x.T.dot(x.dot(theta) - y_train.reshape(-1, 1)))
        theta -= learning_rate * gradient

    return theta.flatten()


def create_plot_for_weights(x, y, x_weights, y_test, title):
    plt.plot(1 / x, y)
    plt.scatter(1 / x_weights, y_test)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title(title)
    plt.show()


data = get_data()
#inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2


# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = create_x_matrix(train_data)

y_test = test_data['MPG'].to_numpy()
x_test = create_x_matrix(test_data)

_, params_number = x_test.shape

# calculate closed-form solution
theta_best = calculate_optimal_theta(x_train, y_train)
print("Theta = ", theta_best)

# calculate error
print("MSE = ", calculate_mse(x_test, y_test, theta_best))

# plot the regression line
x_weights = x_test[:, 1]
x = np.linspace(min(x_weights), max(x_weights), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
create_plot_for_weights(x, y, x_weights, y_test, 'Simple Linear Regression')


# standardization
# 1.15: z = (x - mean) / std
stand_params = dict()
for i in range(1, params_number):
    mean, std = np.mean(x_train[:, i]), np.std(x_train[:, i])
    stand_params[i] = dict({
        'mean': mean,
        'std': std
    })
    x_train[:, i] = (x_train[:, i] - mean) / std
    x_test[:, i] = (x_test[:, i] - mean) / std

# calculate theta using Batch Gradient Descent
theta_best = calculate_gradient_descent_optimal_theta(x_train)
print("Theta = ", theta_best)

# calculate error
print("MSE = ", calculate_mse(x_test, y_test, theta_best))

# plot the regression line
x = np.linspace(min(x_test[:, 1]), max(x_test[:, 1]), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x

# destandardization
for i in range(1, params_number):
    mean = stand_params[i]['mean']
    std = stand_params[i]['std']
    x_test[:, i] = (x_test[:, i] * std) + mean
    if i == 1:
        x = x * std + mean

create_plot_for_weights(x, y, x_test[:, 1], y_test, 'Batch Gradient Descent')
