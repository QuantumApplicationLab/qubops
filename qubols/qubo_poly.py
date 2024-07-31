from sympy.matrices import Matrix, SparseMatrix
import numpy as np
from qubols.encodings import RangedEfficientEncoding
from typing import Optional, Union, Dict
import neal
import dimod
import scipy.sparse as spsp
from .solution_vector import SolutionVector


class QUBO_POLY:

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

    def solve(self, matrices):
        """Solve the linear system

        Args:
            sampler (_type_, optional): _description_. Defaults to neal.SimulatedAnnealingSampler().
            encoding (_type_, optional): _description_. Defaults to RealUnitQbitEncoding.
            nqbit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """

        self.matrices = matrices
        self.num_variables = self.matrices[1].shape[1]

        if not isinstance(self.options["offset"], list):
            self.options["offset"] = [self.options["offset"]] * self.num_variables

        self.solution_vector = self.create_solution_vector()
        self.x = self.solution_vector.create_polynom_vector()

        self.qubo_dict = self.create_qubo_matrix(self.x)

        self.sampleset = self.sampler.sample(
            self.qubo_dict, num_reads=self.options["num_reads"]
        )
        self.lowest_sol = self.sampleset.lowest()
        idx, vars, data = self.extract_data(self.lowest_sol)
        return self.solution_vector.decode_solution(data)

    def create_solution_vector(self):
        """initialize the soluion vector"""
        return SolutionVector(
            size=self.num_variables,
            nqbit=self.options["num_qbits"],
            encoding=self.options["encoding"],
            range=self.options["range"],
            offset=self.options["offset"],
        )

    def create_polynom(self, x: np.ndarray):
        """Creates the polynom from the matrices

        Args:
            x (_type_): _description_
        """
        self.num_equations = self.matrices[1].shape[0]
        polynom = Matrix([0] * self.num_equations)
        for matrix in self.matrices:
            for idx, val in zip(matrix.coords.T, matrix.data):
                polynom[idx[0]] += val * x[idx[1:]].prod()
        return polynom

    def create_qubo_matrix(self, x, strength=100, prec=None):
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

    @staticmethod
    def extract_data(sol):
        """Extracts the data from the solution

        Args:
            sol (_type_): _description_
        """
        # extract the data of the original variables
        idx, vars = [], []
        for ix, s in enumerate(sol.variables):
            if len(s.split("*")) == 1:
                idx.append(ix)
                vars.append(s)

        data = sol.record[0][0]
        return idx, vars, data[idx]
