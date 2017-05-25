import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline

def computeobj(beta, lamb, x, y):
    obj = (1/n)*(np.sum(np.log(1 + np.exp(-y*np.dot(x, beta))))) + lamb*np.sum(beta**2)
    return obj

def computegrad(beta, lamb, x, y):
    grad_beta = -(1/n)*(np.dot(x.T, y/(np.exp(y*np.dot(x, beta)) + 1))) + 2*lamb*beta
    return grad_beta

def backtracking(beta, lamb, x, y, f=computeobj, g=computegrad, eta=1, alpha=0.5, gamma=0.8, max_iter=100):
    grad_beta = g(beta, lamb, x, y)
    norm_grad_beta = np.sqrt(np.sum(grad_beta**2))
    found_eta = 0
    t = 0
    while (found_eta == 0 and t < max_iter):
        if (f(beta - eta*grad_beta, lamb, x, y) < f(beta, lamb, x, y) - alpha*eta*norm_grad_beta**2):
            found_eta = 1
        elif (t == max_iter):
            break
        else:
            eta = eta*gamma
            t += 1
    return eta

def fastgradalgo(beta_init, theta_init, lamb, x, y, max_iter, f=computeobj, g=computegrad):
    beta = beta_init
    theta = theta_init
    eta_init = 1/(max(np.linalg.eigvals(np.dot((1/n)*x.T, x))) + lamb)
    beta_vals = [beta_init]
    t = 0
    while (t < max_iter):
        eta = backtracking(beta, lamb, x, y, eta=eta_init)
        beta_next = theta - eta*g(theta, lamb, x, y)
        theta = beta_next + t*(beta_next - beta)/(t + 3)
        beta = beta_next
        beta_vals.append(beta)
        t += 1
    return beta_vals

spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
spam = spam.dropna()
test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ', header=None)
x = np.asarray(spam.drop(57, axis=1))
y = np.asarray(spam[57])*2 - 1
test_indicator = np.ravel(np.asarray(test_indicator))

x_train = x[test_indicator == 0, :]
x_test = x[test_indicator == 1, :]
y_train = y[test_indicator == 0]
y_test = y[test_indicator == 1]

x_scaler = StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
n = x_train.shape[0]
d = x_train.shape[1]

beta_init = np.zeros(d)
theta_init = np.zeros(d)
fastgradalgo(beta_init, theta_init, 0.1, x_train, y_train, 1000)[-1]

logit = LogisticRegression(C=1/(2*n*0.1), fit_intercept=False, tol=1e-8).fit(x_train, y_train)
logit.coef_