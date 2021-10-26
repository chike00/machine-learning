import numpy as np
import matplotlib.pyplot as plt


fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
fig2, ax3 = plt.subplots()
fig3, ax4 = plt.subplots()

def f(x):
  return x**2

def g(x):
  return x**3

def h(x):
  return x

N = 1000
sets = np.linspace(-10, 10, N)
functionspaceF = f(sets)
functionspaceG = g(sets)
functionspaceH = h(sets)

ax1.set_title("Polynomial curves")
ax1.set_ylabel("y")

ax2.set_xlabel("x")
ax2.set_ylabel("y")

ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("Linear curves")

ax1.plot(sets, functionspaceF, label="f(x) = x^2")
ax2.plot(sets, functionspaceG, label="f(x) = x^3")
ax3.plot(sets, functionspaceH, label="f(x) = x")

ax1.legend()
ax2.legend()
ax3.legend()

def sinx(x):
    return np.sin(x)

N = 10
setSin = np.linspace(-np.pi, np.pi, N)
funcSin = sinx(setSin)

ax4.scatter(setSin, funcSin, c="r")
ax4.plot(setSin, funcSin, "-", c="b", label="f(x)  = sin(x)")
ax4.set_xlabel("x, in radians")
ax4.set_ylabel("y")
ax4.set_title("Scatter graph of sin(x)")
ax4.legend()