#!/usr/bin/env python
"""
A multi variate linear regression model.
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def normal_equation(x, y):
    """
    Computes the closed-form solution to linear regression
    :param X: The input values
    :param y: The output values
    :return: theta params found by normal equation
    """
    return np.linalg.pinv(np.transpose(x) * x) * np.transpose(x) * y


def feature_normalize(x):
    """
    Normalizes the features in x. Normalization improoves the performance of the gradient descent.
    :param x: Features to normalize.
    :return: A normalized version of x where the mean value of each feature is 0 and
    the standard deviation is 1. This is often a good preprocessing step to do when
    working with learning algorithms. The parameters mu and sigma will be used to revert / convert
    the normalized values to non-normalized (original) values.
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = np.divide((x - mu), sigma)
    return x_norm, mu, sigma


def compute_cost(x, y, theta):
    """
    Computes the cost of using theta as the parameter for linear regression
    to fit the data points in X and y.
    :param X: The training set
    :param y: The result set
    :param theta: The parameters
    :return: The cost of using theta to predict y
    """
    m = y.shape[0]  # number of training examples
    h = x * theta  # hypothesis
    errors = (h - y)

    # Calculates the cost function as the average of the squared errors
    j = (1 / (2 * m)) * np.transpose(errors) * errors

    return j[0, 0]


def gradient_descent_multi(x, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta params
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    :param x: The training data
    :param y: The values
    :param theta: The parameters
    :param alpha: The learning rate
    :param num_iters: The number of iterations
    :return: A tuple object where index 0 is the values of theta found by gradient descent
    and index 1 is the history of the cost function for the computed values.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    j_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        h = (x * theta)  # The hypothesis for the given theta params
        errors = (h - y)

        # The slope of the cost function
        derivative_term = (1 / m) * (np.transpose(x) * errors)

        # Simultaneous update
        theta -= alpha * derivative_term

        # Save the cost function for the current iteration.
        # A way to debug if the algorithm is running ok is to plot
        # J_history and check if it is decreasing.
        j_history[iter, 0] = compute_cost(x, y, theta)

    return theta, j_history


if __name__ == '__main__':
    print('Loading data...')
    data = np.loadtxt('data.txt', delimiter=',')
    x = np.matrix(data[:, 0:2])  # input
    y = np.transpose(np.matrix(data[:, 2]))  # output
    m = y.shape[0]  # number of training examples

    # Normalizing features
    print('Normalizing features...')
    x, mu, sigma = feature_normalize(x)

    # Adding the bias term to x. The bias term is the independent term
    # I.e: ax + b, b is the bias term.
    x = np.hstack((np.ones((m, 1)), x))

    # Gradient descent
    print('Running gradient descent...')
    num_iters = 60
    alpha = 1  # The learning rate
    theta = np.zeros((x.shape[1], 1))  # The initial value of theta
    [theta, J_history] = gradient_descent_multi(x, y, theta, alpha, num_iters)

    # You can check if the gradient descent is performing ok
    # through the normal equation. But notice that the normal equation is
    # only suitable for small number of features (inputs). For a large number
    # of features the calculation of the inverse of (x' * x) is too expensive
    # and gradient descent performs better.
    # theta = normal_equation(x,y)  # theta found by normal equation

    # Plot the convergence graph
    print('Plotting the convergence graph...')
    plt.plot(range(0, J_history.shape[0]), J_history, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display the optimal parameters
    print('Theta found: ')
    print(theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.
    size = 1650
    n_bathrooms = 3
    x = np.matrix((size, n_bathrooms))  # input data
    x = np.divide((x - mu), sigma)  # normalized
    x = np.hstack((np.matrix('1'), x))  # bias term
    price = x * theta

    print(f'Predicted price of a {size} sq-ft, {n_bathrooms} br house: ', end = '' )
    print(price[0, 0])
    print()
    print('THE END')
