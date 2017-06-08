import os
from setuptools import setup, find_packages


PACKAGES = find_packages()

opts = dict(name='logistic_reg',
            maintainer='Dane Jordan',
            description='Logistic Regression via Fast Gradient Algorithm',
            url='https://github.com/drjordy66/logistic_reg_fastgrad',
            author='Dane Jordan',
            packages=PACKAGES)

if __name__ == '__main__':
    setup(**opts)
