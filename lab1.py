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
#slopes = np.asarray([0.,-1.,-2.,-3.,-4.,-5.,-6.])
slopes = np.linspace(-8, 0, 50)
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
print("The column for the residuals:\n", residuals)

#Quick and easy way to sum squares of residuals
se = np.sum(np.square(residuals), axis=1)
print("SUM OF SQUARED RESIDUALS:\n", se)

#in future, make this more generalisable, so that you can add as many data
#points as you want -- do this from the get-go

sqRes = np.square(residuals) #squaring the residual
rmse = []
for i in sqRes:
    rmse = np.append(rmse, np.sqrt(np.mean(i)))
print("ROOT MEAN SQUARED ERROR:\n", rmse)

"""
Essentially rmse is where you take the square
of each residual - this is to eliminate the fact
that some residuals and some are negative.
Then you take the mean of them to find generally 
how each prediction curve is deviating from 
its real data point counterpart.

Then you square root because no one likes squares.
"""

"""
Now, I want to be able to generate many slope values,
essentially passing in an entire array of them, and I want to 
be able to get the best fitting one.

To do this, it would be ideal to generalise a lot
of the variables I have above, turning them into
applicable functions
"""