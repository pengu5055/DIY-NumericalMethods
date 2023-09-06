"""
This file should contain the wrapper class for enabling MPI parallelization.
The wrapper class should be able to take a solver as an argument and then
distribute the grid points to the nodes. The nodes should then be able to
solve the PDE in parallel and then gather the solution to the root node.

Currently the wrapper class is implemented in the ColocationSolver class!
Could use improvement in _gather_data() in terms of anti-alising the data
and making sure that the data is in the correct order. Some slight changes
appear at the edges of the individual chunk domains which could probably be
easily mitigated eg. by using a ghost cell approach. 

by: pengu5055
"""
from mpi4py import MPI
import numpy as np
from ColocationSolver import ColocationSolver
from typing import Tuple, Callable, List, Iterable
import socket


class MPI_Node():
    def __init__(self, 
                 solver: ColocationSolver,
                 ) -> None:
        """
        Initialize the MPI node.
        It is better if the number of grid points is divisible
        by the number of nodes. There is a system that should be 
        able to arbitrary number of nodes, but it has not been
        tested rigorously.
        
        Parameters:
            solver: The solver to be used for solving the PDE.

        Returns:
            None
        """
        # Init MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.status = MPI.Status()
        self.name = MPI.Get_processor_name()
        self.hostname = socket.gethostname()
        print(f"Hi! This is rank {self.rank} on {self.hostname}. Ready to go to work...")

        # Init solver
        self.solver = solver
        self.initial_condition = solver.initial_condition
        self.x_range = solver.x_range
        self.N = solver.N
        self.t_points = solver.t_points
        self.D = solver.D
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.N)
        if self.rank == 0:
            self.x_backup = np.copy(self.x)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t_points[1] - self.t_points[0]
        print(f"Rank {self.rank} has initialized the solver.")
    
    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, end - start
        return wrapper
    
    def _distribute_data(self, data):
        """Divide and distribute given data into chunks per rank.

        Surprisingly, this uses no MPI functionality at all. It is
        just a simple list comprehension.
    
        Parameters:
            data: A list of data points which is to be
            divided and distributed.
    
        Returns:
            divided[rank]: A list of data points which is
            assigned to the current rank.
        """
        n = len(data)
        divided = []
        # One x point evolved in time is one data row
        total_data_rows = n
        rows_per_node = total_data_rows // self.size
        extra_rows = total_data_rows % self.size

        self.rows_distribution = [rows_per_node] * self.size

        # This will never go out of range, since the
        # extra rows cannot be more than the number of nodes.
        for i in range(extra_rows):
            self.rows_distribution[i] += 1

        # Distribute the data
        divided = [data[i:i+self.rows_distribution[self.rank]] for i 
                   in range(0, total_data_rows, self.rows_distribution[self.rank])]

        return divided[self.rank]
    
    def _gather_data(self, data, isRoot=False):
        """Gather data from all ranks.
    
        Parameters:
            data: The data to be gathered.
        
        Args:
            isRoot: Whether the current rank is the root rank.
    
        Returns:
            gathered_data: The gathered data.
        """
        gathered_data = self.comm.gather(data, root=0)

        if not isRoot:
            return gathered_data  # comm.gather returns None on non-root ranks
        elif isRoot:
            # ERROR IDENTIFIED: The gathered data is not in the correct order. Was broken by the
            # modifications made for the statistics gatherer.
            output = np.zeros((self.x_backup.shape[0], self.t_points.shape[0]))
            for i in range(self.size):
               output[i*self.rows_distribution[i]:(i+1)*self.rows_distribution[i], :] = gathered_data[i].T

            return output.T
    
    @_internal_function_timer
    def solve(self, method=None, partialOutput=False):
        """
        Parallel solve the PDE but in chunks.

        Parameters:
            method: The method to use for solving the PDE.

        Args:
            partialOutput: Whether to return the partial 
                solution or not.

        Returns:
            solution: The partial solution of the PDE.
            time: The time it took to solve the PDE via
                the @internal_function_timer decorator.
        """
        # Set defaults for method
        if method is None:
            if type(self.solver) == ColocationSolver:
                method = "proper"

        x = self._distribute_data(self.x)
        self.solver._override_x(x)

        if type(self.solver) == ColocationSolver:
            if method == "manual":
                solution, time = self.solver.solve_Manually()
            elif method == "proper":
                solution, time = self.solver.solve_Properly()
        else:
            raise TypeError("The solver must be of type ColocationSolver!")
        
        if partialOutput:
            # Copy of the solution pre-gather
            self.partialSolution = np.copy(solution)

        if self.rank != 0:
            self.solution = self._gather_data(solution)
        elif self.rank == 0:
            self.solution = self._gather_data(solution, isRoot=True)
        else:
            # Should never happen.
            raise ValueError("The rank must be an integer.")
        
        if partialOutput:
            return self.solution, self.partialSolution
        else:
            # Return the partial solution more as a proof of concept
            return self.solution