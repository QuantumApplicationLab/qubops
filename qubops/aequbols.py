import numpy as np
from qubops.encodings import RangedEfficientEncoding
from typing import Optional, Union, Dict
import neal

from .qubops import qubops


class AEqubops(qubops):
    """Linear solver using an adaptative encoding qubops

    Args:
        qubops (_type_): _description_
    """

    def __init__(self, options: Optional[Union[Dict, None]] = None):
        """Linear Solver using adaptative encoding QUBO

        Args:
            options: dictionary of options for solving the linear system
        """

        default_solve_options = {
            "sampler": neal.SimulatedAnnealingSampler(),
            "encoding": RangedEfficientEncoding,
            "range": 1.0,
            "offset": 0.0,
            "num_qbits": 11,
            "iterations": 3,
            "temperature": 1000,
            "num_reads": 100,
            "verbose": False,
        }
        print(options)
        self.options = self._validate_solve_options(options, default_solve_options)
        self.sampler = self.options.pop("sampler")

        if self.options["encoding"] != RangedEfficientEncoding:
            raise ValueError("AEqubops is only possible with RangedEfficientEncoding")

    def solve(self, matrix: np.ndarray, vector: np.ndarray):
        """Solve the linear system

        Args:
            sampler (_type_, optional): _description_. Defaults to neal.SimulatedAnnealingSampler().
            encoding (_type_, optional): _description_. Defaults to RealUnitQbitEncoding.
            nqbit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        if not isinstance(matrix, np.ndarray):
            matrix = matrix.todense()

        self.A = matrix
        self.b = vector
        self.size = self.A.shape[0]

        if not isinstance(self.options["offset"], list):
            self.options["offset"] = [self.options["offset"]] * self.size

        for _iiter in range(self.options["iterations"]):
            print(_iiter, self.options["offset"], self.options["range"])
            self.solution_vector = self.create_solution_vector()
            self.x = self.solution_vector.create_polynom_vector()
            self.qubo_dict = self.create_qubo_matrix(self.x)

            self.sampleset = self.sampler.sample_qubo(
                self.qubo_dict, num_reads=self.options["num_reads"]
            )

            self.update_encoding()

        self.lowest_sol = self.sampleset.lowest()
        return self.solution_vector.decode_solution(self.lowest_sol.record[0][0])

    def update_encoding(self):
        """_summary_"""

        count = np.array([r[2] for r in self.sampleset.record])
        energy = np.array([r[1] for r in self.sampleset.record])
        energy -= energy.min()
        energy = np.exp(-self.options["temperature"] * energy)

        res = np.array(
            [self.solution_vector.decode_solution(r[0]) for r in self.sampleset.record]
        )

        # weighted average to compute the new offset values
        average = np.average(res, axis=0, weights=count * energy).tolist()

        # std to compute the new range
        std = np.std(res, axis=0).tolist()
        std = [s if s > 1e-3 else 1.0 for s in std]

        self.options["range"] = std
        self.options["offset"] = average
