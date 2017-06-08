"""
This code shows an example implementing the fast gradient algorithm to compute
the beta coefficients and misclassification error. It compares the beta
coefficients to that of sklearn.
"""


import src.logistic_reg as lreg
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# create simulated dataset
np.random.seed(0)

data1 = np.random.random(size=(250, 50)) - 0.05
data2 = np.random.random(size=(300, 50)) + 0.05
data = np.concatenate((data1, data2), axis=0)

label1 = np.ones(shape=(250, ))
label2 = np.ones(shape=(300, ))*-1
label = np.concatenate((label1, label2), axis=0)

# define the split between train and test data
x_train, x_test, y_train, y_test = train_test_split(data,
                                                    label,
                                                    random_state=0)

# standardize the data
x_scaler = StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
n = x_train.shape[0]
d = x_train.shape[1]

# initialize the beta and theta values
beta_init = np.zeros(d)
theta_init = np.zeros(d)

# run the fast gradient algorithm to find the beta coefficients
fastgrad_betas = lreg.fastgradalgo(beta_init=beta_init,
                                   theta_init=theta_init,
                                   lamb=0.1,
                                   x=x_train,
                                   y=y_train,
                                   max_iter=1000)

# run sci-kit learn's LogisticRegression() to find the beta coefficients
logit = LogisticRegression(C=1/(2*n*0.1),
                           fit_intercept=False,
                           tol=1e-8).fit(x_train, y_train)

# print the coefficients found using the fast gradient algorithm and sklearn
print("\nFast Gradient Algorithm Coefficients:\n", fastgrad_betas[-1])
print("\nSci-kit Learn's LogisticRegression() Coefficients:\n", logit.coef_)

# apply the coefficients found using the fast gradient algorithm to test set
y_predict = (np.dot(x_test, fastgrad_betas[-1]) > 0)*2 - 1

# print the misclassification error
print("\nMisclassification Error: %.2f%%" % (np.mean(y_predict != y_test)*100))
