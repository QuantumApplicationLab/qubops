from sympy.matrices import Matrix
import numpy as np
from typing import Optional, Union, Dict
import neal
import dimod
from .qubo_poly import QUBO_POLY
from .mixed_solution_vector import MixedSolutionVector


class QUBO_POLY_MIXED(QUBO_POLY):

    def __init__(
        self,
        mixed_solution_vectors: MixedSolutionVector,
        options: Optional[Union[Dict, None]] = None,
    ):
        """Polynomial of degree 2 Solver using QUBO

        Solve the following equation

        ..math:
            P_0 + P_1 x + P_2 x \otimes x = 0

        Args:
            options: dictionary of options for solving the linear system
        """

        default_solve_options = {
            "sampler": neal.SimulatedAnnealingSampler(),
            "num_reads": 100,
            "verbose": False,
        }
        self.options = self._validate_solve_options(options, default_solve_options)
        self.sampler = self.options.pop("sampler")
        self.mixed_solution_vectors = mixed_solution_vectors

    def create_bqm(self, matrices, strength=10):
        """Create the bqm from the matrices

        Args:
            matrices (tuple): matrix of the system

        Returns:
            dimod.bqm: binary quadratic model
        """

        self.matrices = matrices
        self.num_variables = self.matrices[1].shape[1]

        self.x = self.mixed_solution_vectors.create_polynom_vector()
        self.extract_all_variables()

        return self.create_qubo_matrix(self.x, strength=strength)

    def sample_bqm(self, bqm):
        """Sample the bqm"""

        return self.sampler.sample(bqm, num_reads=self.options["num_reads"])

    def decode_solution(self, solution):
        """_summary_

        Returns:
            _type_: _description_
        """
        idx, vars, data = self.extract_data(solution)
        return self.mixed_solution_vectors.decode_solution(data)

    def solve(self, matrices):
        """Solve the linear system

        Args:
            sampler (_type_, optional): _description_. Defaults to neal.SimulatedAnnealingSampler().
            encoding (_type_, optional): _description_. Defaults to RealUnitQbitEncoding.
            nqbit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """

        # create the bqm
        self.qubo_dict = self.create_bqm(matrices)

        # sample the bqm
        self.sampleset = self.sample_bqm(self.qubo_dict)
        self.lowest_sol = self.sampleset.lowest()

        # sample the systen and return the solution
        return self.decode_solution(self.lowest_sol)

    def extract_all_variables(self):
        """Extracs all the variable names and expressions"""
        self.all_vars = []
        self.all_expr = []
        for var in self.x:
            expr = [(str(k), v) for k, v in var.as_coefficients_dict().items()]
            self.all_expr.append(expr)
            self.all_vars += [str(k) for k in var.as_coefficients_dict().keys()]

    def create_polynom(self, x: np.ndarray):
        """Creates the polynom from the matrices

        Args:
            x (_type_): _description_
        """
        self.num_equations = self.matrices[1].shape[0]
        polynom = Matrix([0] * self.num_equations)

        for imat, matrix in enumerate(self.matrices):

            for idx, val in zip(matrix.coords.T, matrix.data):
                if imat == 0:
                    polynom[idx[0]] += val
                else:
                    polynom[idx[0]] += val * x[idx[1:]].prod()
        return polynom

    def create_qubo_matrix(self, x, strength=10, prec=None):
        """Create the QUBO dictionary requried by dwave solvers
        to solve the polynomial equation P0 + P1 x + P2 x x = 0


        Args:
            Anp (np.array): matrix of the linear system
            bnp (np.array): righ hand side of the linear system
            x (sympy.Matrix): unknown

        Returns:
            _type_: _description_
        """

        polynom = self.create_polynom(np.array(x))

        polynom = polynom.T @ polynom

        polynom = polynom[0]
        polynom = polynom.expand()
        polynom = polynom.as_ordered_terms()
        polynom = self.create_poly_dict(polynom, prec=prec)
        bqm = dimod.make_quadratic(polynom, strength=strength, vartype=dimod.BINARY)

        return bqm

    @staticmethod
    def create_poly_dict(polynom, prec=None):
        """Creates a dict from the sympy polynom

        Args:
            polynom (_type_): _description_

        Returns:
            Dict: _description_
        """
        out = dict()

        for term in polynom:
            m = term.args
            if len(m) == 0:
                continue

            if len(m) == 2:
                varname = str(m[1])
                tmp = varname.split("**")
                if len(tmp) == 1:
                    exponent = 1
                else:
                    varname, exponent = tmp
                    exponent = int(exponent)
                key = (varname,) * exponent

            elif len(m) > 2:
                key = tuple()
                for mi in m[1:]:
                    mi = str(mi)
                    tmp = mi.split("**")
                    if len(tmp) == 1:
                        key += (tmp[0],)
                    if len(tmp) == 2:
                        varname = tmp[0]
                        exp = int(tmp[1])
                        key += (varname,) * exp

            if key not in out:
                out[key] = 0.0

            out[key] += m[0]

        if prec is None:
            return out

        elif prec is not None:
            nremoved = 0
            out_cpy = dict()
            for k, v in out.items():
                if np.abs(v) > prec:
                    out_cpy[k] = v
                else:
                    nremoved += 1
            print("Removed %d elements" % nremoved)
            return out_cpy

    def extract_data(self, sol):
        """Extracts the data from the solution

        Args:
            sol (_type_): _description_
        """
        # extract the data of the original variables
        idx, vars = [], []
        for ix, s in enumerate(sol.variables):
            if s in self.all_vars:
                idx.append(ix)
                vars.append(s)

        data = sol.record[0][0]
        return idx, vars, data[idx]
