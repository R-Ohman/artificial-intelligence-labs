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
def create_x_matrix(data):
    weight = 1 / data['Weight'].to_numpy()
    cylinders = data['Cylinders'].to_numpy()
    horsepower = 1/ data['Horsepower'].to_numpy()
    displacement = 1 / data['Displacement'].to_numpy()
    model_year = data['Model Year'].to_numpy()
    ones = np.ones_like(weight)
    return np.c_[ones, weight]


y_train = train_data['MPG'].to_numpy()
x_train = create_x_matrix(train_data)

y_test = test_data['MPG'].to_numpy()
x_test = create_x_matrix(test_data)

_, params = x_test.shape

# TODO: calculate closed-form solution
def get_theta_best(x, y):
    # o - observations
    # 1.13: (o.T * o)^-1 * o.T * y
    inv_product = np.linalg.inv(np.matmul(x.T, x))
    return np.matmul(np.matmul(inv_product, x.T), y)


theta_best = get_theta_best(x_train, y_train)
print("Theta = ", theta_best)


# TODO: calculate error
def get_mse(x_arr, y_arr, theta):
    size = x_arr.shape[0]
    mse = 0
    for i in range(0, size):
        prediction = theta[0]
        for j in range(1, len(theta)):
            prediction += theta[j] * x_arr[i][j]
        mse += (y_arr[i] - prediction) ** 2

    return mse/size


print("MSE = ", get_mse(x_test, y_test, theta_best))

x_test_weight = x_test[:, 1]
# plot the regression line
x = np.linspace(min(x_test_weight), max(x_test_weight), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
#y = 1 / y

plt.plot(1/x, y)
plt.scatter(1/x_test_weight, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Simple Linear Regression')
plt.show()

# TODO: standardization
# 1.15: z = (x - mean) / std
for i in range(1, params):
    mean = np.mean(x_train[:, i])
    std = np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - mean) / std
    x_test[:, i] = (x_test[:, i] - mean) / std


# TODO: calculate theta using Batch Gradient Descent
def get_gradient_theta(x):
    theta = np.random.rand(1, params).reshape(-1, 1)  # random initialization of N x 1 matrix
    learning_rate = 0.01

    train_size = x.shape[0]
    for i in range(100000):
        gradient = (2 / train_size) * (x.T.dot(x.dot(theta) - y_train.reshape(-1, 1)))
        theta -= learning_rate * gradient

    return theta.flatten()


theta_best = get_gradient_theta(x_train)
print("Theta = ", theta_best)

# TODO: calculate error
print("MSE = ", get_mse(x_test, y_test, theta_best))

# plot the regression line

x = np.linspace(min(x_test[:, 1]), max(x_test[:, 1]), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x

# destandardization
for i in range(1, params):
    mean = np.mean(x_train[:, i])
    std = np.std(x_train[:, i])
    x_test[:, i] = (x_test[:, i] * std) + mean
    if i == 1:
        x = (x * std) + mean

plt.plot(x, y)
plt.scatter(x_test[:, 1], y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
