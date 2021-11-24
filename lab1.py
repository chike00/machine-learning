# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:43:18 2021

@author: chike
"""

import numpy as np
import matplotlib.pyplot as plt

def linear_simple(x):
    return -2.5*x

Xtrain1 = np.linspace(-1,1,3)
#training set, 3 values between -1 and 1

np.random.seed(10) #repeatable randomness

n3 = np.random.normal(0,1.0,3) #mean, standard deviation, number of datapoints

ytrain1 = linear_simple(Xtrain1) + n3 #application across matrix

"""plot"""
plt.scatter(Xtrain1, ytrain1)

wlist = np.asarray([0., -1., -2., -3., -4., -5., -6.])
X = np.linspace(-1, 1, 50)
plt.scatter(Xtrain1, ytrain1, marker='X')
for i in range(len(wlist)):
    plt.plot(X, wlist[i]*X)

"""residuals"""
for w in wlist:
    print(w*Xtrain1 - ytrain1)
print("RESIDUALS FOR WLIST\n\n")
"""
This isn't completely intuitive at first.
But since vector operations will do it over the
entire vector first, I don't need to use 
nested loops or anything like that. Do it for 
'one' gradient, and it will compute it for each datapoint
"""

wrange = np.linspace(-6,0,6)
for w in wrange:
    print(w*Xtrain1 - ytrain1)
print("RESIDUALS FOR WRANGE\n\n")
    
"""loss"""
#for wlist:
for w in wlist:
    print(np.sum((w*Xtrain1 - ytrain1)**2))
print("SQUARED RESIDUALS FOR WLIST\n\n")

#for wrange:
for w in wrange:
    print(np.sum((w*Xtrain1 - ytrain1)**2))
print("SQUARED RESIDUALS FOR WRANGE\n\n")

"""INFO:
    
If we have multiple datasets modelling the same
thing but they have different sizes, then we can
take an average so that size isn't important

Remember that we squared we did the some of the
squares as well. In order to get a value with
the same magnitude as the predicted values, we
can take a square root.
"""

"""MSE - MEAN SQUARED ERROR FOR ONLY ONE GRADIENT"""
def mse(w, x, y):
    return np.mean((np.multiply(w, x) - y)**2)

"""L1E - ABSOLUTE VALUE (this still gets rid of the fact
that you have negative and positive deviations from the
predicted
value"""
def l1e(w, x, y):
    return np.mean(np.abs(np.multiply(w,x) - y))

"""RMSE - ROOT MEAN SQUARED ERROR FOR ONLY ONE GRADIENT"""
def rmse(w, x, y):
    return np.mean(np.sqrt((np.multiply(w, x) - y)**2))

print("Mean squared errors:")
for w in wrange:
    print(mse(w,Xtrain1, ytrain1))

print("Absolute values:")
for w in wrange:
    print(l1e(w, Xtrain1, ytrain1))

"""MSE FOR ARRAY OF GRADIENTS"""
def ar_mse(w,x,y):
    return np.array([mse(wi,x,y) for wi in w])
"""
This is a really concise way of saying for each
slope, run our mse function which takes 1 slope,
and then put it all in one array and return it
"""

def ar_rmse(w,x,y):
    return np.array([rmse(wi,x,y) for wi in w])

def ar_l1e(w,x,y):
    return np.array([l1e(wi,x,y) for wi in w])

"""GRAPHING THE LOSS USING THESE THREE METHODS"""

wlims = np.linspace(-8,0,50)

fig, (ax1,ax2,ax3) = plt.subplots(figsize=(15,4), nrows=1, ncols=3)

ax1.plot(wlims, ar_mse(wlims, Xtrain1, ytrain1), label="mean SQUARED error")
ax1.legend()

ax2.plot(wlims, ar_l1e(wlims, Xtrain1, ytrain1), label="mean ABSOLUTE error")
ax2.legend()

ax3.plot(wlims, ar_rmse(wlims, Xtrain1, ytrain1), label="RMSE")
ax3.legend()

"""
argmin() will treat the entire array/vector as one flat array and just return the minimum value
argmin(axis=0) will return an array representing the minimum values for EACH column
argmin(axis=1) will return an array representing the minimum values for EACH row

I think since out ar_mse function returns a vector
which does have multiple rows, argmin just returns
the index which produces the minimum value

https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

While it's not directly related, this explains
how numpy arrays are shaped and how we can alter our VIEW of that shape

https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
In the examples section it shows how the use
of the optional axis parameter changes it's
behaviour
"""

#MSE
wbest, least_error = (wlims[np.argmin(ar_mse(wlims,Xtrain1, ytrain1))],np.min(ar_mse(wlims,Xtrain1, ytrain1)))
print(wbest, least_error) # the best-fit value and the corresponding error 

#RMSE
wbest2, least_error2 = (wlims[np.argmin(ar_rmse(wlims,Xtrain1, ytrain1))],np.min(ar_rmse(wlims,Xtrain1, ytrain1)))
print(wbest2, least_error2) # the best-fit value and the corresponding error 


"""Question:
Should the values for wbest and least_error be the same for rmse instead of mse?

Hypothesis:
I think so. Given rmse is just mse where all the values have been square rooted,
I can't see why either would be different. 

Outcome:
While wbest WAS the same, least_error was different.
Additionally, when you increase the amount of slopes
and hence the precision, even the wbest differs. Odd.

When number of slopes was 5:
-4.0 0.18068276777718004
-4.0 0.3096975874600519

When number of slopes was 50:
-3.9183673469387754 0.1784307650181058
-3.9183673469387754 0.3096975874600519

When number of slopes was 10000:
-3.938793879387939 0.17816078658665715
-4.045204520452045 0.3096975874600519
"""

"""Trying to match our predicted line to the 
dataset"""

fig2, ax4 = plt.subplots()
"""
X = np.linspace(-2,2,50)
ax4.plot(X, wbest*X, label="best-fit")
ax4.scatter(Xtrain1,ytrain1,c='r',marker='X', label='training set')
ax4.plot(X, linear_simple(X), label="true line")
ax4.legend()

OLD CODE. SEE BELOW FOR NEWER VERSION.
"""

"""We find that the machine learned line
is actually (for now) a better fit than the true
function. I think this goes to show how a
dataset with a small sample can be misleading.

Next, we'll generate more points to see whether our trained
slope still keeps its accuracy.
"""

npts = 10
Xtrain2 = np.linspace(-1,1,npts)
noise = np.random.normal(0,1.0,npts)
ytrain2 = linear_simple(Xtrain2) + noise  # generate some more points from the same function

X = np.linspace(-2,2,10)
ax4.plot(X,wbest*X, label='best-fit')
ax4.plot(X,-2.5*X, '--', c='k', label='true-fn')
ax4.scatter(Xtrain1,ytrain1,c='r',marker='X', label='training set')
ax4.scatter(Xtrain2,ytrain2,c='g',marker='X', label='additional pts')

wbest3, least_error3 = (wlims[np.argmin(ar_rmse(wlims,Xtrain2, ytrain2))],np.min(ar_rmse(wlims,Xtrain2, ytrain2)))
print(wbest3, least_error3)

"""
-2.612244897959184 0.4553772993069729

Now we see that by having more points, our
machine learned line isn't as accurate.
It highlights the need to calculate loss over
many points.

We could increase the precision (and risk
overtraining) by increasing the number of points,
or increasing the number of gradients
"""

ax4.plot(X, wbest3*X, label="new best-fit")

"""Gradient descent:"""

npts = 20
Xtrain3 = np.linspace(-1,1,npts)
noise = np.random.normal(0,1.0,npts)
ytrain3 = linear_simple(Xtrain3) + noise
#our new training set

fig3, ax5 = plt.subplots()

def loss_slope_w1(w1, Xtrain, ytrain):
    return (2/len(Xtrain))*(np.dot(w1*Xtrain - ytrain, Xtrain))

"""
IMPORTANT
_________

Make sure you review your book for why you are
dotting these things together.

x^2 is the same as dot(x,x) for example 
"""

wlims = np.linspace(-6,2,50) # Generate 50 slope values for the straight line models
ax5.plot(wlims,ar_mse(wlims, Xtrain3, ytrain3), label='loss-function')
ax5.scatter([-5.],ar_mse([-5.],Xtrain3, ytrain3),c='r')

gw = loss_slope_w1(-5.0, Xtrain3, ytrain3)
loss = mse(-5., Xtrain3, ytrain3)
"""
gw is the gradient of the curve at x=5

loss is kinda telling us what we can already see.
We can see on the curve that at -5, the loss on the y-axis
is ~4.
"""

ax5.plot(wlims[:15],gw*(wlims[:15]+5) + loss, label='grad of loss') # plotting the slope using the first few values of w (prettifying )
"""
Here is a kinda sneaky way of plotting the gradient line
We use the first 15 points from -6 to 2,
same as the original axis. And then we apply
the gradient function on it. We cap it
to the first 15 points because if not
it would kinda shrink the importance of the
loss curve. I'm not sure why we add 5 to it
however.
"""
ax5.set_title('Mean square loss function and the tangent at w=0.5')
ax5.legend()


print("slope of loss fn = ",loss_slope_w1( -5.0, Xtrain3, ytrain3), \
      ", mse loss fn = ", mse(-5., Xtrain3, ytrain3))

print("\n==========================\n")
    
"""
Here, we've taken the derivative
of the average squared loss function in the
function loss_slope_w1
"""

def gradientdescent0(initialweight, X, y, rate, numiter):
    whistory = []
    msehistory = [] 
    w = initialweight
    for i in range(numiter): 
        loss = mse(w, X, y)
        whistory.append(w)
        msehistory.append(loss)
        grad = loss_slope_w1(w, X, y)
        w = w - rate*grad  # go a certain distance opposite to the slope (downward) 
    return w, np.asarray(whistory), np.asarray(msehistory)
"""
initialweight - You want to perform gradient descent starting from some random value
X - X training set
y - y training set
rate - 
numiter - number of iterations

So we have a whistory - which is the next weight (slope) it has iterated to
"""

#for i in range(1000):
#    print('numiter = {}:\n'.format(i), gradientdescent0(-5., Xtrain3, ytrain3, .2, i), "\n\n")

i=1000
wbest4 = gradientdescent0(-5, Xtrain3, ytrain3, 0.2, i)[1][i-1]

ax4.plot(X, wbest4*X, label="after gradient descent")

inclusive = 50

fig4, ax6 = plt.subplots()
XX = np.int64(np.linspace(1,inclusive,inclusive))

#ax6.plot(XX, gradientdescent0(-5, Xtrain3, ytrain3, 0.2, i)[2][i-1])

ax6.plot(XX, np.array([[gradientdescent0(-5, Xtrain3, ytrain3, 0.1, j)[2][j-1]] for j in XX]))
ax6.set_xlabel("Number of iterations")
ax6.set_ylabel("Loss")
ax6.set_title("Number of iterations necessary to minimise loss")

"""
This function applies only to linear regression.
See Rogers and Girolami, equations 1.8 and 1.10 for mathematical form and
derivation.

X is the data points, y is true value

w1 was 2.5 
"""
def linear_fit(X,y):
    num = (X*y).mean() - (X.mean()*y.mean())
    den = (X**2).mean() - (X.mean())**2
    w1 = num/den
    w0 = y.mean() - w1*X.mean()
    return w0, w1

w0 = linear_fit(X, linear_simple(X))[0]
w1 = linear_fit(X, linear_simple(X))[1]

ax4.plot(X, w1*X + w0, label="Linear reg equation")

print(w1)
ax4.legend()
"""
This was cool.
What we did was use the generator feature we saw earlier to change the number of iterations.
By making the  the gradient descent 
"""

"""
The essence of machine learning seems to
be having a dataset, creating loss functions
and having a means to calculate the loss,
and then trying out loads of slopes
"""