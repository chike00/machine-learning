# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:40:20 2021

@author: chike
"""

import numpy as np
import matplotlib.pyplot as plt

# men's 100m sprint times
olympics100m = np.asarray([1900,11,
1904,11,
1906,11.2,
1908,10.8,
1912,10.8,
1920,10.8,
1924,10.6,
1928,10.8,
1932,10.3,
1936,10.3,
1948,10.3,
1952,10.4,
1956,10.5,
1960,10.2,
1964,10,
1968,9.95,
1972,10.14,
1976,10.06,
1980,10.25,
1984,9.99,
1988,9.92,
1992,9.96,
1996,9.84,
2000,9.87,
2004,9.85,
2008,9.69,
2012,9.63,
2016,9.81]);

# women's 100m sprint times
olympics100f=np.asarray([
1948, 11.90,
1952, 11.50,
1956, 11.50,
1960, 11.00,
1964, 11.40,
1968, 11.08,
1972, 11.07,
1976, 11.08,
1980, 11.06,
1984, 10.97,
1988, 10.54,
1992, 10.82,
1996, 10.94,
2000, 10.75,
2004, 10.93,
2008, 10.78,
2012, 10.75,
2016, 10.71])

"""
Here we use the closed form solution to the derivative of loss function of 
the using the minimum square error equation.

They were discussing here:
https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution
that using this method isn't always good. I'll need to remember gradient 
descent.

A closed form solution is one which uses "standard" operations (subjective)
and an answer (whether function or number) can be arrived at in finite steps.

The equation:
    x = SUM(1 to INFINITY){
        x^2
    }

is not a closed form solution, as it requires infinite additions.
"""

X = olympics100m[::2]
y = olympics100m[1::2]

fig1, ax1 = plt.subplots()
ax1.scatter(X, y)
ax1.plot(X, y, label="Men's olympic times")

def linear_fit(X,y):
    num = (X*y).mean() - (X.mean()*y.mean())
    den = (X**2).mean() - (X.mean())**2
    w1 = num/den
    w0 = y.mean() - w1*X.mean()
    return w0, w1

w0 = linear_fit(X, y)[0]
w1 = linear_fit(X, y)[1]

ax1.plot(X, w1*X + w0, label="closed form best fit")

ax1.legend()