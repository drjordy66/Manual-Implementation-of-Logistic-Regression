Examples
========

This directory contains two examples on:
- A simulated dataset
- A real-world example dataset

Both examples import/create the datasets within the example. They implement the fast gradient algorithm in the src folder to solve the logisitc classification problem and compare the results to those of sci-kit learn's [LogisticRegression()](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The .ipynb files are the same as the .py, but allow visualization of the training process.

After installing `logistic_reg_fastgrad`, run python and enter the following command:

```
from src import logistic_reg as lreg
```

This will import all of the functions included in the package and may be called by entering:

```
lreg.<function_name>
```