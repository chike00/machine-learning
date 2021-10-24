import numpy as np
import matplotlib.pyplot as plt

fig1, (ax1, ax2) = plt.subplots(nrows=2)
fig2, ax3 = plt.subplots()

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

fig1.title("Polynomial curves")
fig2.title("Linear curves")

ax1.legend()
ax2.legend()
ax3.legend()

ax1.plot(sets, functionspaceF, label="f(x) = x^2", shareX=True)
ax2.plot(sets, functionspaceG, label="f(x) = 0")
ax3.plot(sets, functionspaceF, label="f(x) = x")