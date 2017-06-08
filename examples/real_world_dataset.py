import src.logistic_reg as lreg
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# load dataset and drop NAs
spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/'
                     'datasets/spam.data', sep=' ', header=None)
spam = spam.dropna()
test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/'
                               'ElemStatLearn/datasets/spam.traintest',
                               sep=' ', header=None)

# declare data and labels
x_data = np.asarray(spam.drop(57, axis=1))
y_data = np.asarray(spam[57])*2 - 1
test_indicator = np.ravel(np.asarray(test_indicator))

# define the split between train and test data
x_train = x_data[test_indicator == 0, :]
x_test = x_data[test_indicator == 1, :]
y_train = y_data[test_indicator == 0]
y_test = y_data[test_indicator == 1]

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
                                   max_iter=1000)[-1]

# run sci-kit learn's LogisticRegression() to find the beta coefficients
logit = LogisticRegression(C=1/(2*n*0.1),
                           fit_intercept=False,
                           tol=1e-8).fit(x_train, y_train)

# print the coefficients found using the fast gradient algorithm and sklearn
print(fastgrad_betas)
print(logit.coef_)
