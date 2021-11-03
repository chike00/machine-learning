import matplotlib.pyplot as plt
import numpy as np

def linear_simple(x):
    return -2.5*x
Xtrain1 = np.linspace(-1,1,3)
np.random.seed(10) 
# a seed ensures that re-running the random number generator yields the same outcome
# this is helpful for the purpose of sanity checking your implementations
n3 = np.random.normal(0,1.0,3)
ytrain1 = linear_simple(Xtrain1) + n3
# Check to see what the data looks like 
print("x= ",Xtrain1,"y= ", ytrain1,"noise added= ", n3)
plt.scatter(Xtrain1,ytrain1, marker='X')


"""
Pay attention to this:
    
Look how efficiently we plotted several plots, we simply
iterated through the slopes array, and applied a function 
across an entire dataset. Probably only works for product based
functions
"""

#list of slopes to try
slopes = np.asarray([0.,-1.,-2.,-3.,-4.,-5.,-6.])
xslopes = np.linspace(-1, 1, 50)
for i in range(len(slopes)):
    plt.plot(xslopes,slopes[i]*xslopes, label=slopes[i])

plt.legend() #we know which colour corresponds to the function

#former ways of computing residuals
"""
for i in range(len(slopes)):
    print("For slope: ", slopes[i])
    for j in range(len(Xtrain1)):
        print("residual ", j, " = ", (Xtrain1[j] * slopes[i]) - ytrain1[j])
"""        
"""
for po in range(len(slopes)):
    print("For slope: ", slopes[po])
    print((Xtrain1 * slopes[po]) - ytrain1)
"""

"""
I'm going to dot the slopes and the Xtrain together.
To do this, I need to put them in the correct shapes. For this, 
I use newaxis.
"""

slopes = slopes[:, np.newaxis]
#print(np.shape(slopes))

Xtrain1 = Xtrain1[np.newaxis, :]
#print(np.shape(Xtrain1))

sx = np.dot(slopes, Xtrain1)
#print(sx)

#numpy understands what you mean when you tell it to subtract another matrix
#it will do this for every line. Which is cool.
residuals = sx - ytrain1
print("The column for the residuals: \n", residuals)

#Quick and easy way to sum squares of residuals
mse = np.sum(residuals**2, axis=1)
print(mse)