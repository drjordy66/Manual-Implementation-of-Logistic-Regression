logistic_reg_fastgrad
=====================

This code implements the fast gradient algorithm to solve the following logistic regression problem:

![logistic regression](https://github.com/drjordy66/logistic_reg_fastgrad/blob/master/images/problem.PNG "logistic regression")

See the [examples](https://github.com/drjordy66/logistic_reg_fastgrad/tree/master/examples) section on how to use this algorithm on both a real-world dataset and a simulated dataset for classification problems.

All data will be downloaded during the example scripts. No outside downloads are required.

__PLEASE NOTE__: This algorithm does not implement cross-validation to find the optimal regularization parameter.

Installation
------------

To install `logistic_reg_fastgrad` you will need to clone the repository on your computer using the following `git` command:

```
git clone https://github.com/drjordy66/logistic_reg_fastgrad.git
```

Next, to install the package you will need to go into the cloned directory and run the `setup.py` file:

```
cd logistic_reg_fastgrad/
python setup.py install
```