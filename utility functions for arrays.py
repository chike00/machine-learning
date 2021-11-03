# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 16:39:28 2021

@author: chike
"""

a = [1,2,3,4,5]

"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html

NumPy arrays

slices/subarrays
steps
reverse slices
reverse stepping

multi dimensional:
selecting x rows and y columns
reversing

taking slices of arrays and modifying them
actually modifies the real thing

choosing certain views, 2 rows and 2 columns for example
or 2 rows, 4 columns view, take every other row

array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])


>>> a[::2, ::2] 
>>> array([[1, 3],
       [7, 9]])

to take a copy of an np-array and to ensure that modifying 
it does not change its parent, you use array.copy()

np.arange(1,20,2) returns [1,3,5,7,...19] a 1-d array

.reshape(m,n) takes, provided the number of elements inside is equal,
and turns it into whatever shape you want

np.random.normal(mean, number of standard deviations, number of values)

"""

