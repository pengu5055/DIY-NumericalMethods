"""
This file will contain the class for the colocation method. 
The colocation method is a numerical method for solving PDE's.
It is a spectral method, meaning that it uses a Fourier basis to approximate the solution.
We're going to need to solve matrix equations, so we'll use numpy.

by: pengu5055
"""
import numpy as np
from typing import Tuple, Callable, Iterable
from scipy.sparse import diags
from scipy.linalg import solve_banded

class ColocationSolver:
    def __init__(self,
                 initial_condition: Callable[[float], float],
                 x_range: Tuple[float, float],
                 N: int,
                 t_points: Iterable[float],
                 D: float,
                 ) -> None:
        """
        Initialize the solver with a grid size N in the ranges set by x_range.
        Set the initial condition and the diffusion constant D. Prepare to solve
        the PDE at the time points given by t_points.

        Parameters:
            initial_condition: The initial condition of the PDE in
                the form of a function of x. In this case for the
                Heat equation, it is the temperature at t=0.
            x_range: The boundaries of the x grid for which the
                solution is calculated.
            N: The number of grid points in the x direction.
            t_points: The time points at which the solution is calculated.
                (The number of M points is inferred from the length of 
                this array.)
            D: The diffusion constant.
        """
        self.initial_condition = initial_condition
        self.x_range = x_range
        self.N = N
        self.t_points = t_points
        self.D = D
        self.x = np.linspace(self.x_range[0], self.x_range[1], N)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t_points[1] - self.t_points[0]
    
    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, start - end
        return wrapper
    
    def _override_x(self, x):
        """
        Override the x grid with a new one. This is due to the fact that MPI
        will distribute the grid points to the nodes. The nodes will need to
        know the grid points they are responsible for but the wrapper class
        will take the already initialized solver as an argument. Therefore,
        the nodes will need to override the x grid.

        Parameters:
            x: The new x grid.
        """
        self.x = x
        self.x_range = (x[0], x[-1])
        self.dx = self.x[1] - self.x[0]
        self.N = len(x)

    @_internal_function_timer
    def solve_Manually(self) -> np.ndarray:
        """
        This method of solving the PDE uses the colocation method but 
        is implemented manually. It is generally FASTER than the built-in
        method! At least for smaller grid sizes!

        It will gather all required data from the object itself and solve
        the PDE. The solution is stored in the object itself but is also
        returned as well as the time it took to solve the PDE via the
        @internal_function_timer decorator.

        Parameters:
            None - all parameters are set in the __init__ method.

        Returns:
            solution_m: The solution to the PDE at each time point.
        """
        self.solution_m = np.zeros((len(self.t_points), self.N))
        f_init_vec = np.zeros(self.N)
        f_init_vec = self.initial_condition(self.x)
        c_init_vec = np.zeros(self.N)

        # Create tridiagonal matricies of coefficients
        A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()
        B = ((6*self.D)/self.dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        A_inv = np.linalg.inv(A)

        # 1. Solve: A * c_init_vec = f_init_vec
        c_init_vec = self._do_TDMA(A, f_init_vec)

        # 2. Solve: A * dc/dt = B * c -> dc/dt = (c[i+1] - c[i]) / dt  <= Backward Euler
        c = np.zeros((len(self.t_points), self.N))
        c[0] = c_init_vec
        for i in range(1, len(self.t_points)):
            c[i] = A_inv*B*self.dt @ c[i-1] + c[i-1]
        
        # 3. Solve: A * c = f
        self.solution_m = (A @ c.T).T

        return self.solution_m
    
    @_internal_function_timer
    def solve_Properly(self) -> np.ndarray:
        """
        This method of solving the PDE uses bulit-in functions from scipy.
        It is generally SLOWER than the manual method! At least for smaller
        grid sizes.

        It will gather all required data from the object itself and solve
        the PDE. The solution is stored in the object itself but is also
        returned as well as the time it took to solve the PDE via the 
        @internal_function_timer decorator.

        Parameters:
            None - all parameters are set in the __init__ method.

        Returns:
            solution: The solution to the PDE at each time point.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        f_init_vec = np.zeros(self.N)
        f_init_vec = self.initial_condition(self.x)
        c_init_vec = np.zeros(self.N)

        # Create tridiagonal matricies of coefficients
        # A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()
        diagonals = [[1] * (self.N - 1), [4] * self.N, [1] * (self.N - 1)]
        A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        # Convert the tridiagonal matrix to a dense banded matrix
        A_banded = np.zeros((3, self.N))
        A_banded[0] = np.pad(diagonals[0], (1, 0), mode="constant")
        A_banded[1] = np.pad(diagonals[1], (0, 0), mode="constant")
        A_banded[2] = np.pad(diagonals[2], (0, 1), mode="constant")
        A_inv = np.linalg.inv(A)

        # (np.round(np.linalg.inv(A_inv))) <- Rounding may help stop propagation of errors due to 
        # floating point precision (non zero values in the inverse matrix)

        B = ((6*self.D)/self.dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        # 1. Solve: A * c_init_vec = f_init_vec
        c_init_vec = solve_banded((1, 1), A_banded, f_init_vec)
        
        # 2. Solve: A * dc/dt = B * c -> dc/dt = (c[i+1] - c[i]) / dt  <= Forward Euler
        c = np.zeros((len(self.t_points), self.N))
        c[0] = c_init_vec
        for i in range(1, len(self.t_points)):
            c[i] = A_inv*B*self.dt @ c[i-1] + c[i-1]

        # 3. Solve: A * c = f
        self.solution = (A @ c.T).T

        return self.solution

    def _do_TDMA(self, A, f):
        """
        Do the Thomas algorithm for a tridiagonal matrix A and a vector f.
        Essentially solving: A * c = f for an unknown vector c.

        Parameters:
            A: The tridiagonal matrix.
            f: The vector of function values.

        Returns:
            c_new: The solution vector.
        """
        alpha = np.zeros(self.N)
        beta = np.zeros(self.N)
        c_new = np.zeros(self.N)

        alpha[1] = A[0, 1] / A[0, 0]
        beta[1] = f[0] / A[0, 0]

        for i in range(1, self.N-1):
            den = (A[i, i] - A[i, i-1] * alpha[i])
            alpha[i+1] = A[i, i+1] / den
            beta[i+1] = (f[i] - A[i, i-1] * beta[i]) / den
        
        c_new[self.N-1] = (f[self.N-1] - A[self.N-1, self.N-2] * beta[self.N-1]) / (A[self.N-1, self.N-1] - A[self.N-1, self.N-2] * alpha[self.N-1])
        for i in range(self.N-2, -1, -1):
            c_new[i] = beta[i+1] - alpha[i+1] * c_new[i+1]
        
        return c_new