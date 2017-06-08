"""
Minimizing Logistic Regression using Fast Gradient Algorithm with Backtracking
"""


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def computeobj(beta, lamb, x, y):
    """
    Compute a return a single value for the objective function
    :param beta: array
        initialize betas (typically zeros)
    :param lamb: float
        regularization parameter
    :param x: array
        data
    :param y: array
        labels
    """
    
    n = x.shape[0]

    obj = (1/n)*(np.sum(np.log(1 + np.exp(-y*np.dot(x, beta))))) \
        + lamb*np.sum(beta**2)

    return obj


def computegrad(beta, lamb, x, y):
    """
    Compute the gradient of the objective function and return a vector size d
    :param beta: array
        initialize betas (typically zeros)
    :param lamb: float
        regularization parameter
    :param x: array
        data
    :param y: array
        labels
    """

    n = x.shape[0]

    grad_beta = -(1/n)*(np.dot(x.T, y/(np.exp(y*np.dot(x, beta)) + 1))) \
        + 2*lamb*beta

    return grad_beta


def backtracking(beta, lamb, x, y, eta=1, alpha=0.5, gamma=0.8, max_iter=100):
    """
    :param beta: array
        initialize betas (typically zeros)
    :param lamb: float
        regularization parameter
    :param x: array
        data
    :param y: array
        labels
    :param eta: float
        initial step-size for backtracking
    :param alpha: float
        constant for sufficient decrease condition
    :param gamma: float
        constant for sufficient decrease condition
    :param max_iter: int
        stopping criterion
    """

    grad_beta = computegrad(beta, lamb, x, y)
    norm_grad_beta = np.sqrt(np.sum(grad_beta**2))
    found_eta = 0
    t = 0

    while found_eta == 0 and t < max_iter:
        if (computeobj(beta - eta*grad_beta, lamb, x, y) <
                computeobj(beta, lamb, x, y) - alpha*eta*norm_grad_beta**2):
            found_eta = 1
        elif t == max_iter:
            break
        else:
            eta = eta*gamma
            t += 1

    return eta


def fastgradalgo(beta_init, theta_init, lamb, x, y, max_iter):
    """
    :param beta_init: array
        initialize betas (typically zeros)
    :param theta_init: array
        initialize thetas (typically zeros)
    :param lamb: float
        regularization parameter
    :param x: array
        data
    :param y: array
        labels
    :param max_iter: int
        stopping criterion
    """

    n = x.shape[0]
    beta = beta_init
    theta = theta_init
    eta_init = 1/(max(np.linalg.eigh(np.dot((1/n)*x.T, x))[0]) + lamb)
    beta_vals = [beta_init]
    t = 0

    while t < max_iter:
        eta = backtracking(beta, lamb, x, y, eta=eta_init)
        beta_next = theta - eta*computegrad(theta, lamb, x, y)
        theta = beta_next + t*(beta_next - beta)/(t + 3)
        beta = beta_next
        beta_vals.append(beta)
        t += 1

    return beta_vals
