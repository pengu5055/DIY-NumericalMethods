"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
from ColocationSolver import ColocationSolver


# Test new spectral solver
def initial_condition(x):
    return np.sin(2*np.pi*x*10)

# Solve for these points
t = np.linspace(0, 0.01, 1000)

# Solve for this grid range
x_range = (0, 1)

# Solve for this grid size
N = 1000

# Solve for this diffusion constant
D = 1e-5

# Initialize solver
solver = ColocationSolver(initial_condition, x_range, N, t, D)

# Solve the PDE
T, t = solver.solve_Manually()

# Plot the solution
plt.plot(solver.x, T[0])  # Initial condition
plt.plot(solver.x, T[N//2])  # Middle condition
plt.show()