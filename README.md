logistic_reg_fastgrad
=====================

This code implements the fast gradient algorithm to solve the following $\ell_2^2$-regularized logistic regression problem:

$$\min_{\beta\epsilon\mathbb{R}}F(\beta):=\frac{1}{n}\sum_{i=1}^nlog\left(1+\exp(-y_ix_i^T\beta)\right)+\lambda\lvert\lvert\beta\rvert\rvert_2^2$$

See the [[examples](https://github.com/drjordy66/logistic_reg_fastgrad/tree/master/examples)] section on how to use this algorithm on both a real-world dataset and a simulated dataset for classification problems.