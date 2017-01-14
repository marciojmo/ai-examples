#!/usr/bin/env python
"""
A multi variate logistic regression model.
"""

import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot as plt


def sigmoid(z):
    """
    Computes the sigmoid function of z. [1 / (1+e^-z)]
    :param z: The activation value(s)
    :return: The sigmoid function of z.
    """
    g = np.divide(1, (1 + np.power(np.e, -z)))
    return g


def cost_function(x, y, theta):
    """
    Compute cost and gradient for logistic regression
    Computes the cost of using theta as the parameter for logistic regression.
    This function also returns the gradient of the cost function in order to
    use a 3rd party optimization function.
    :param x: The training data
    :param y: The output of this training data
    :param theta: The theta params
    :return: The cost and gradient for the given theta
    """
    m = y.shape[0]  # number of training examples
    h = sigmoid(x * theta)  # hypothesis
    # logistic cost using the sigmoid function
    j = (-np.transpose(y) * np.log(h) - np.transpose(1 - y) * np.log(1 - h)) / m
    # logistic gradient for the given step (partial derivative)
    grad = (np.transpose(x) * (h - y)) / m

    return j, grad


if __name__ == '__main__':
    print('Loading data...')
    data = np.loadtxt('data.txt', delimiter=',')
    x = np.matrix(data[:, 0:-1])  # input
    y = np.transpose(np.matrix(data[:, -1]))  # output
    m = y.shape[0]  # number of training examples

    # Plotting data
    print('Plotting data...')
    pos = np.where( y == 1 )
    neg = np.where( y == 0 )

    plt.plot( x[pos,0], x[pos,1], 'r+', label='Admitted' )
    plt.plot( x[neg,0], x[neg,1], 'bo', label='Not admitted' )
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    #plt.legend(['Admitted', 'Not Admitted'])

    plt.show()
