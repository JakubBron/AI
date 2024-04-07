import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

def makeObservationMatrix(matrix):
    matrixOfOnes = np.ones((matrix.shape))
    return np.hstack((matrixOfOnes, matrix))


# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy().reshape(-1, 1)
x_train = train_data['Weight'].to_numpy().reshape(-1, 1)

y_test = test_data['MPG'].to_numpy().reshape(-1,1)
x_test = test_data['Weight'].to_numpy().reshape(-1,1)

print()

# TODO: calculate closed-form solution
""" formula: theta {matrix} = (X.T * X)^-1 * X.T * y
where X - matrix m * (n+1) of observations, like [1, x_1_1, x_1_2, ... x_1_n
                                                  |    \      \          | 
                                                  1, x_m_1, x_m_2, ... x_m_n]
theta - matrix (n+1)x1 of parameters, like [theta_0, theta_1, ... theta_n]
y - matrix m*1 of outputs, like [y_1, y_2, ... y_m]
In our case, observations size a.k.a n = 1, m = number of rows in input data.
So, dataset needs to be reshaped!
"""

# theta_best = [0, 0]
# m - observationMatrix
m = makeObservationMatrix(x_train)
theta_best = np.linalg.inv(m.T.dot(m)).dot(m.T).dot(y_train)
#print("[DEBUG] Theta best is: ", theta_best.flatten())

###################################################################################################################
# TODO: calculate error
"""
error (MSE) = 1/m * SUM (from i=1 to m) (y_real_i - y_predicted_i)^2
where m = number of observations (number of rows in x_train / x_test)
y_predicted_i = theta_best[0] + theta_best[1] * x_i
"""
mse = np.sum( (y_train - (theta_best[0] + theta_best[1] * x_train)) ** 2 )
mse = mse / len(x_train)
print(f'[DEBUG] MSE error is: {mse}, b, a (for y = b + ax, from theta) is {theta_best.flatten()}')

###################################################################################################################
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

###################################################################################################################
# TODO: standardization
"""
Here, we use Z standardization, which is:
z = (x - mean) / std
where x - input data, mean - mean of input data, std - standard deviation of input data
"""
x_train_mean = np.mean(x_train)         # !!! Należy pamiętać że w celu obliczenia średniej i odchylenia standardowego nie należy wykorzystywać zbioru treningowego ze względu na ryzyko wycieku danych testowych (testing dataset leakage).
x_train_std = np.std(x_train)           # !!! j.w - w instrukcji błąd?
standarized_x_train = (x_train - x_train_mean) / x_train_std
standarized_x_train = standarized_x_train.reshape(-1, 1)

y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
standarized_y_train = (y_train - y_train_mean) / y_train_std
standarized_y_train = standarized_y_train.reshape(-1, 1)

standarized_x_test = (x_test - x_train_mean) / x_train_std
standarized_x_test = standarized_x_test.reshape(-1, 1)
standarized_y_test = (y_test - y_train_mean) / y_train_std
standarized_y_test = standarized_y_test.reshape(-1,1)

#print(f'[DEBUG] TRAIN: x_mean: {x_train_mean}, x_std: {x_train_std}, y_mean: {y_train_mean}, y_std: {y_train_std}')
#print(f'[DEBUG] standarized TRAIN: x_mean: {np.mean(standarized_x_train)}, x_std: {np.std(standarized_x_train)}, y_mean: {np.mean(standarized_y_train)}, y_std: {np.std(standarized_y_train)}')
#print(f'[DEBUG] standarized TEST: x_mean: {np.mean(standarized_x_test)}, x_std: {np.std(standarized_x_test)}, y_mean: {np.mean(standarized_y_test)}, y_std: {np.std(standarized_y_test)}')

###################################################################################################################
# TODO: calculate theta using Batch Gradient Descent
"""
1. Start with random theta from (0,1) range
2. Calculate gradient of MSE (mean squared error) with respect to theta: 
    gradient_MSE = 2/m * X.T * (X * theta - y), m - number of observations
3. Theta := Theta - learning_rate * gradient_MSE
4. Jump to 2 and repeat until convergence (iterations > 0)
"""

theta = np.random.rand(2, 1)
DEBUG_prev_theta = theta
iterations = 10000
learning_rate = 0.1
matrix = makeObservationMatrix(standarized_x_train)
m = len(standarized_x_train)
for i in range(iterations):
    gradient_MSE = 2*matrix.T.dot(matrix.dot(theta) - standarized_y_train) / m
    theta = theta - learning_rate * gradient_MSE
    #if i % 500 == 0: print(f'[DEBUG] Iteration {i}, Theta: {theta.flatten()}, prevTheta / theta: {(DEBUG_prev_theta/theta).flatten()}')

print(f'[DEBUG] Theta from Batch Gradient Descent: {theta.flatten()}')

###################################################################################################################
# TODO: calculate error
"""
error (MSE) = 1/m * SUM (from i=1 to m) (y_real_i - y_predicted_i)^2
where m = number of observations (number of rows in x_train / x_test)
y_predicted_i = theta_best[0] + theta_best[1] * x_i
"""
# m = len(x_train)
mse = np.sum( (standarized_y_train - (theta[0] + theta[1] * standarized_x_train)) ** 2 )
mse = mse / len(standarized_x_train)
print(f'[DEBUG] MSE error is: {mse}, b, a (for y = b + ax, from theta) is {theta.flatten()}')

###################################################################################################################
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()