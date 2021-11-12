import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10) 

def linear_simple(x):
    return -2.5*x
    """When you change this, ensure that it also lies
    within your variable range also """

def calc_residuals(wlist, x, y):
    #y = linear_simple(x) + n3    
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

numberOfSlopes = 100
slopes = np.linspace(-8,8,numberOfSlopes) #(min, max, number of slopes)

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

slopeSTAR = slopes[np.argmin(rmse)]

print("The slope which gives the minimum value is:\ny =", slopeSTAR, "x")

ax1.plot(Xtrain1, Xtrain1*slopeSTAR)

"""
to solve this, maybe get the index/key of the minimum rmse 
- we know roundabouts what the correct answer is because
we've given the function for it.
"""

print("The associated rmse/loss of this is:\n", np.min(rmse))

"""
GRADIENT DESCENT
________________

"""
print("\n=================\n")

fig3, (ax3, ax4) = plt.subplots(nrows=2)

gradient = -5

npts = 20
Xtrain3 = np.linspace(-1,1,npts)
noise = np.random.normal(0,1.0,npts)
ytrain3 = linear_simple(Xtrain3) + noise
ax3.scatter(Xtrain3, ytrain3)
ax3.plot(Xtrain3, Xtrain3*[gradient], c='g')
ax3.set_title("Mean square loss function")
"""Handles dataset plotting"""

wlims = np.linspace(-6,2,50)
rmse2 = rootMeanSquareResiduals(calc_residuals(wlims, Xtrain3, ytrain3))
ax4.plot(wlims, rmse2)
ax4.scatter([-5.], rootMeanSquareResiduals(calc_residuals(np.array([-5]), Xtrain3, ytrain3)), c='r')
"""Shows how the loss varies with the slope"""

def loss_slope_w1(w1, Xtrain, ytrain):
    return (2/len(Xtrain))*(np.dot(w1*Xtrain - ytrain, Xtrain))

print("Slope of loss fn = ", loss_slope_w1(gradient, Xtrain3, ytrain3),
      ", mse loss fn = ", rootMeanSquareResiduals(calc_residuals(np.array([gradient]), Xtrain3, ytrain3)))

gw = loss_slope_w1(gradient, Xtrain3, ytrain3) 
loss = rootMeanSquareResiduals(calc_residuals(np.array([-5]), Xtrain3, ytrain3))
ax4.plot(wlims[:15],gw*(wlims[:15]+5) + loss, label='grad of loss') # plotting the slope using th