"""
Test the MPI_Node class.

NOTICE!
Has to be run from the command line in the following way:

mpiexec -n [number of processes] python -m mpi4py MPI_demo.py

or similarly for mpirun:

mpirun -np [number of processes] python -m mpi4py MPI_demo.py
"""
from mpi import MPI_Node
import numpy as np
from ColocationSolver import ColocationSolver


# Test new parallel wrapper for solver
def gaussian_Initial(x, a=5, sigma=1):
        return np.exp(-((-x+a)/2)**2 / sigma**2)

# Solve for these points
t = np.linspace(0, 0.15, 1000)

# Solve for this grid range
x_range = (-0.5, 10.5)

# Solve for this grid size
N = 1000

# Solve for this diffusion constant
D = 1e-3

# Initialize solver
solver = ColocationSolver(gaussian_Initial, x_range, N, t, D)

# Initialize MPI node
node = MPI_Node(solver)
node.solve(method="manual")


# Plot the solution (a little higher level example)
import matplotlib.pyplot as plt
import matplotlib as mpl
if node.rank == 0:
    # x_backup since x is overwritten in the MPI_Node class
    # but in the root rank a backup is kept
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=np.min(node.solution), vmax=np.max(node.solution))
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(node.solution)))

    for i, sol in enumerate(node.solution):
        # Invert index to get the correct colors
        # -1 otherwise first solution is plotted on top
        plt.plot(node.x_backup, sol, color=cmap[-i - 1])

    scalar_Mappable = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    
    cb = plt.colorbar(scalar_Mappable, ax=ax, label=r"$T\>[arb. units]$",
                      orientation="vertical")
    ticks = np.round(-1*cb.ax.get_yticks(), 2)  # *-1 to get the correct values otherwise they are inverted
    cb.ax.set_yticklabels(ticks)
    ax.set_xlabel(r"$x\>[arb. units]$")
    ax.set_ylabel(r"$t\>[arb. units]$")
    plt.suptitle("Evolution of the solution solved by MPI Colocation method")
    plt.show()
