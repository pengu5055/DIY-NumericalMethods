"""
This is the source script for the difference method partial differential equation solver.
Demonstrated here it is used to solve Schrodinger's equation in 1D for a particle trapped 
in a harmonic oscillator potential. The difference method is a subset of the the finite
difference methods.

by: pengu5055
"""
import numpy as np
from typing import Callable, Tuple, Iterable


class FDMSolver():
    def __init__(self,
                 initial_condition: Callable[[np.ndarray], np.ndarray],
                 potential: Callable[[np.ndarray], np.ndarray],
                 x_range: Tuple[float, float],
                 t_points: Iterable[float],
                 N: int,
                 ) -> None:
        """
        Initialize the solver. 

        Parameters:
            initial_condition: The initial condition of the PDE in
                the form of a function of x. In this case for the
                Schrodinger equation, it is the wavefunction at t=0.
            potential: The potential for the Hamiltonian in the form
                of a function of x. 
            x_range: The boundaries of the x grid for which the
                solution is calculated.
            t_points: The time points at which the solution is
                calculated. (The number of M points is inferred from
                the length of this array.)
            N: The number of grid points in the x direction.

        Returns:
            None

        """
        self.initial_condition = initial_condition
        self.potential = potential
        self.x_range = x_range
        self.t = t_points
        self.N = N
        self.x = np.linspace(x_range[0], x_range[1], N)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        # Initialize the solution array
        self.solution = np.empty((len(self.t), self.N), dtype=np.complex128)

    def solve(self) -> np.ndarray:
        """
        Solve the PDE iteratively using the difference method to
        calculate the wavefunction at each time point. Essentially
        just matrix multiplication.

        Parameters:
            None - all parameters are set in the __init__ method.

        Returns:
            solution: The solution to the PDE at each time point.
        """
        # Set the initial condition
        self.solution[0] = self.initial_condition(self.x)
        
        # Set the boundary conditions
        # I think this might be a potential error. It should be 0 across all indices
        self.solution[:, 0] = 0
        self.solution[:, -1] = 0

        # Set the potential
        self.V = self.potential(self.x)

        # Set up the matrix
        self.b = 1j*(self.dt/(2*self.dx**2))
        self.a = - self.b/2
        self.d = 1 + self.b + 1j * self.dt/2 * self.V

        self.A = np.diag(self.a * np.ones(self.N-1), -1) + \
            np.diag(self.d * np.ones(self.N), 0) + \
            np.diag(self.a * np.ones(self.N-1), 1)

        self.A_inv = np.linalg.inv(self.A)
        self.A_dagger = np.conj(self.A).T

        # Solve the PDE
        for i in range(1, len(self.t)):
            self.solution[i] = self.A_inv @ self.A_dagger @ self.solution[i-1]
        
        return self.solution