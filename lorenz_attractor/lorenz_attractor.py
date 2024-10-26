import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Lorenz system parameters
def lorenz(X, t, sigma=10, rho=28, beta=8/3):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial conditions
X0 = [0.1, 1.0, 1.05]
t = np.linspace(0, 50, 10000)
X = odeint(lorenz, X0, t)

# Plotting
plt.plot(X[:, 0], X[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
