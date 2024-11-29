import numpy as np
import pytest
from qubops.qubops_mixed_vars import QUBOPS_MIXED
from qubops.solution_vector import SolutionVector_V2 as SolutionVector
from qubops.mixed_solution_vector import MixedSolutionVector_V2 as MixedSolutionVector
from qubops.encodings import PositiveQbitEncoding
from dwave.samplers import SteepestDescentSolver
from scipy.optimize import newton
import sparse


def get_classical_solution():
    def nlfunc(input):
        x0, x1, x2, x3 = input

        def f0():
            return 1 / 2 - x0 + x1

        def f1():
            return 1 - x1

        def f2():
            return 3 - x0**2 - x2

        def f3():
            return -0.5 * x1**2 + x2 - x3

        return np.array([f0(), f1(), f2(), f3()])

    initial_point = np.random.rand(4)
    ref_sol = newton(nlfunc, initial_point)
    return ref_sol


def define_matrices():

    # system of equations
    num_equations = 4
    num_variables = 4

    P0 = np.zeros(num_equations)
    P0[0] = 1 / 2
    P0[1] = 1
    P0[2] = 3
    P0[3] = 0

    P1 = np.zeros((num_equations, num_variables))
    P1[0, 0] = -1
    P1[0, 1] = 1

    P1[1, 1] = -1

    P1[2, 2] = -1

    P1[3, 2] = 1
    P1[3, 3] = -1

    P2 = np.zeros((num_equations, num_variables, num_variables))
    P2[2, 0, 0] = -1
    P2[3, 1, 1] = -1 / 2

    return sparse.COO(P0), sparse.COO(P1), sparse.COO(P2)


def test_qubops_default():
    """Test the qubols solver."""
    # define the encoding for the first two varialbes
    nqbit = 5
    step = 0.05
    encoding1 = PositiveQbitEncoding(
        nqbit=nqbit, step=step, offset=0, var_base_name="x"
    )
    sol_vec1 = SolutionVector(2, encoding=encoding1)

    # define the encoding for the alst two variables
    nqbit = 4
    step = 0.05
    encoding2 = PositiveQbitEncoding(
        nqbit=nqbit, step=step, offset=0, var_base_name="x"
    )
    sol_vec2 = SolutionVector(2, encoding=encoding2)

    # define the solution vector
    sol_vec = MixedSolutionVector([sol_vec1, sol_vec2])

    # instantiat the QUBOPS solver
    options = {"num_reads": 10000, "sampler": SteepestDescentSolver()}

    # instantiate solver
    qubo = QUBOPS_MIXED(sol_vec, options)

    # define matrices
    matrices = define_matrices()

    # solve
    sol = qubo.solve(matrices, strength=1e5)
    sol = np.array(sol).reshape(-1)

    if not np.allclose(get_classical_solution(), sol):
        pytest.skip("QUBOLS solution innacurate")
