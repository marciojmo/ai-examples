#!/usr/bin/env python
"""
A multi variate logistic regression model using scipy fmin_cg
"""

import numpy as np
from scipy.optimize import fmin_bfgs, fmin_cg, minimize
from matplotlib import pyplot as plt


def sigmoid(z):
    """
    Computes the sigmoid function of z. [1 / (1+e^-z)]
    :param z: The activation value(s)
    :return: The sigmoid function of z.
    """
    g = np.divide(1, (1 + np.power(np.e, -z)))
    return g


def cost_function(theta, x, y):
    """
    Compute cost and gradient for logistic regression
    Computes the cost of using theta as the parameter for logistic regression.
    Obs: Theta must be the first parameter due to the optimization function
    :param theta: The theta params
    :param x: The training data
    :param y: The output of this training data
    :return: The cost and gradient for the given theta
    """
    m, n = x.shape  # number of training examples and features
    theta = np.reshape(theta, (n, 1))  # we need to reshape theta
    h = sigmoid(x * theta)  # hypothesis
    # logistic cost using the sigmoid function
    j = (-np.transpose(y) * np.log(h) - np.transpose(1 - y) * np.log(1 - h)) / m
    # logistic gradient for the given step (partial derivative)
    # grad = (np.transpose(x) * (h - y)) / m
    return j


def predict(x,theta):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta.
    Computes the predictions for X using a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    :param x: The input data
    :param theta: The params
    :return: The class of the input
    """
    m, n = x.shape  # number of training examples and features
    theta = np.reshape(theta, (n, 1))  # we need to reshape theta
    z = x * theta
    return z >= 0

if __name__ == '__main__':
    print('Loading data...')
    data = np.loadtxt('data.txt', delimiter=',')
    x = np.matrix(data[:, 0:-1])  # input
    y = np.transpose(np.matrix(data[:, -1]))  # output
    m, n = x.shape

    # Plotting data
    print('Plotting data...')
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(x[pos, 0], x[pos, 1], 'r+', label='Admitted')
    plt.plot(x[neg, 0], x[neg, 1], 'bo', label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(('Admitted', 'Not Admitted'))
    plt.show()

    # Adds the extra term
    x = np.hstack((np.ones((m, 1)), x))

    # Optimizes the parameters
    initial_theta = np.zeros(n + 1)
    theta = fmin_cg(cost_function, initial_theta, args=(x, y))

    # Predicts the class of an arbitrary element (should be 0/False according to the training set)
    x_test = np.matrix([1,0.96141,0.085526 ])
    print(predict(x_test,theta))


