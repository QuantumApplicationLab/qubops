from sympy.matrices import Matrix, SparseMatrix
import numpy as np
from qubols.encodings import RangedEfficientEncoding
from typing import Optional, Union, Dict
import neal
import dimod
import scipy.sparse as spsp
from .solution_vector import SolutionVector


class QUBO_POLY2:

    def __init__(self, options: Optional[Union[Dict, None]] = None):
        """Polynomial of degree 2 Solver using QUBO

        Solve the following equation

        ..math:
            P_0 + P_1 x + P_2 x \otimes x = 0

        Args:
            options: dictionary of options for solving the linear system
        """

        default_solve_options = {
            "sampler": neal.SimulatedAnnealingSampler(),
            "encoding": RangedEfficientEncoding,
            "range": 1.0,
            "offset": 0.0,
            "num_qbits": 11,
            "num_reads": 100,
            "verbose": False,
        }
        self.options = self._validate_solve_options(options, default_solve_options)
        self.sampler = self.options.pop("sampler")

    @staticmethod
    def _validate_solve_options(
        options: Union[Dict, None], default_solve_options: Dict
    ) -> Dict:
        """validate the options used for the solve methods

        Args:
            options (Union[Dict, None]): options
        """
        valid_keys = default_solve_options.keys()

        if options is None:
            options = default_solve_options

        else:
            for k in options.keys():
                if k not in valid_keys:
                    raise ValueError(
                        "Option {k} not recognized, valid keys are {valid_keys}"
                    )
            for k in valid_keys:
                if k not in options.keys():
                    options[k] = default_solve_options[k]

        return options

    def solve(
        self, matrix_p0: np.ndarray, matrix_p1: np.ndarray, matrix_p2: np.ndarray
    ):
        """Solve the linear system

        Args:
            sampler (_type_, optional): _description_. Defaults to neal.SimulatedAnnealingSampler().
            encoding (_type_, optional): _description_. Defaults to RealUnitQbitEncoding.
            nqbit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        for matrix in [matrix_p0, matrix_p1, matrix_p2]:
            if not isinstance(matrix, np.ndarray):
                matrix = matrix.todense()

        self.P0 = matrix_p0
        self.P1 = matrix_p1
        self.P2 = matrix_p2

        self.size = self.P1[0]

        if not isinstance(self.options["offset"], list):
            self.options["offset"] = [self.options["offset"]] * self.size

        self.solution_vector = self.create_solution_vector()
        self.x = self.solution_vector.create_polynom_vector()

        self.qubo_dict = self.create_qubo_matrix(self.x)

        self.sampleset = self.sampler.sample_qubo(
            self.qubo_dict, num_reads=self.options["num_reads"]
        )
        self.lowest_sol = self.sampleset.lowest()

        return self.solution_vector.decode_solution(self.lowest_sol.record[0][0])

    def create_solution_vector(self):
        """initialize the soluion vector"""
        return SolutionVector(
            size=self.size,
            nqbit=self.options["num_qbits"],
            encoding=self.options["encoding"],
            range=self.options["range"],
            offset=self.options["offset"],
        )

    def create_qubo_matrix(self, x, prec=None):
        """Create the QUBO dictionary requried by dwave solvers
        to solve the polynomial equation P0 + P1 x + P2 x x = 0

        A x = b

        Args:
            Anp (np.array): matrix of the linear system
            bnp (np.array): righ hand side of the linear system
            x (sympy.Matrix): unknown

        Returns:
            _type_: _description_
        """

        for matrix in [self.P0, self.P1]:
            if isinstance(matrix, spsp.spmatrix):
                matrix = SparseMatrix(*matrix, dict(matrix.todok().items()))
            else:
                matrix = Matrix(matrix)

        self.P2 = [Matrix(matrix) for matrix in self.P2]

        polynom = self.P0 + self.P1 @ x

        x2 = x @ x.T
        for ip, p2 in enumerate(self.P2):
            polynom[ip] += np.sum(p2 @ x2)

        polynom = polynom.T @ polynom

        polynom = polynom[0]
        polynom = polynom.expand()
        polynom = polynom.as_ordered_terms()
        polynom = self.create_poly_dict(polynom, prec=prec)
        bqm = dimod.make_quadratic(polynom, strength=5, vartype=dimod.BINARY)

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
            print(m)
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
