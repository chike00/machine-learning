import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10) 

def linear_simple(x):
    return 2*x

def calc_residuals(wlist, x, y):
    y = linear_simple(x) + n3    
    wlist = wlist[:, np.newaxis]
    x = x[np.newaxis, :]
    sx = np.dot(wlist, x)
    residuals = sx - y
    return residuals

def sumSquaresOfResiduals(listOfResiduals):
    return np.sum(np.square(listOfResiduals), axis=1)

def rootMeanSquareResiduals(listOfResiduals):
    sqRes = np.square(listOfResiduals)
    rmse = []
    for i in sqRes:
        rmse = np.append(rmse, np.sqrt(np.mean(i)))
    return rmse

xmin = -8
xmax = 1
dataPoints = 20

Xtrain1 = np.linspace(xmin,xmax,dataPoints)

n3 = np.random.normal(0,1.0,dataPoints)
ytrain1 = linear_simple(Xtrain1) + n3

#print("x= ",Xtrain1,"y= ", ytrain1,"noise added= ", n3)

numberOfSlopes = 10
slopes = np.linspace(-8,3,numberOfSlopes) #(min, max, number of slopes)

"""
More slopes means a more accurate loss function - more to check
"""

xslopes = np.linspace(xmin, xmax, 2) #the x values we're operating on

"""
setting the final argument to 1 means that we draw a line 
that is bound between just one point. It's basically a dot.
"""

fig1, ax1 = plt.subplots()

"""
TOGGLE OFF THIS COMMENT IF YOU WANT TO SEE ALL THE SLOPES
PLOTTED ON THE GRAPH
________________________________

for i in range(len(slopes)):
    ax1.plot(xslopes,slopes[i]*xslopes, label=slopes[i]) #for the actual lines

ax1.legend()
"""

ax1.scatter(Xtrain1,ytrain1, marker='X') #for the X's

"""
xslopes is the range of x's

slopes[i]*xslopes essentially applies the 
gradients forming what we see on screen, but this is independent
of the scatter points
"""

ssr = sumSquaresOfResiduals(calc_residuals(slopes, Xtrain1, ytrain1))
rmse = rootMeanSquareResiduals(calc_residuals(slopes, Xtrain1, ytrain1))

#print("Sum square of residuals:\n", ssr)
print("Root mean square error:\n", rmse)

fig2, ax2 = plt.subplots()
ax2.plot(slopes, rmse)

print("The slope which gives the minimum value is:\ny =", "?", "x")

"""
to solve this, maybe get the index/key of the minimum rmse 
- we know roundabouts what the correct answer is because
we've given the function for it.
"""

print("The associated rmse/loss of this is:\n", np.min(rmse))