from sympy.matrices import Matrix
import numpy as np
from copy import deepcopy
from .encodings import RangedEfficientEncoding


class MixedSolutionVector(object):

    def __init__(self, solution_vectors):
        """init the mixed solution vector

        Args:
            solution_vectors (List): A list of SolutionVector instances
        """
        self.solution_vectors = solution_vectors
        self.nqbit = []
        self.idx_start_data = self.get_indexes_data()
        self.encoded_reals = self.create_encoding()

    def create_encoding(self):
        """Create the enconding for all the unknowns


        Returns:
            list[RealEncoded]:
        """
        encoded_reals = []

        idx_vars = 0
        for sol_vec in self.solution_vectors:
            is_ranged_encoding = sol_vec.encoding == RangedEfficientEncoding
            for i in range(sol_vec.size):
                var_base_name = sol_vec.base_name + "_%03d" % (idx_vars + 1)
                idx_vars += 1

                if is_ranged_encoding:
                    args = (
                        sol_vec.nqbit,
                        sol_vec.range[i],
                        sol_vec.offset[i],
                        var_base_name,
                    )
                else:
                    args = (sol_vec.nqbit, var_base_name)
                encoded_reals.append(sol_vec.encoding(*args))
                self.nqbit.append(sol_vec.nqbit)
        return encoded_reals

    def get_indexes_data(self):
        """Get the indices of the start/end of the data for each encoding

        Returns:
            np.array: lis of start/end index
        """
        idx_start_data = [0]
        for sv in self.solution_vectors:
            idx_start_data.append(sv.nqbit * sv.size)
        idx_start_data[-1] += 1
        return np.cumsum(idx_start_data)

    def create_polynom_vector(self):
        """Create the list of polynom epxressions

        Returns:
            sympy.Matrix: matrix of polynomial expressions
        """
        pl = []
        for real in self.encoded_reals:
            pl.append(real.create_polynom())
        return Matrix(pl)

    def decode_solution(self, data):
        """Decode the solution

        Args:
            data (list): data from the annealer

        Returns:
            list: decoded numbers
        """
        sol = []
        for iv, sv in enumerate(self.solution_vectors):
            idx_start = self.idx_start_data[iv]
            idx_end = self.idx_start_data[iv + 1]
            sv_data = data[idx_start:idx_end]
            sol.append(list(sv.decode_solution(sv_data)))
            idx_start += sv.size
        return sol


class MixedSolutionVector_V2(MixedSolutionVector):

    def __init__(self, solution_vectors):
        """init the mixed solution vector

        Args:
            solution_vectors (List): A list of SolutionVector instances
        """
        self.solution_vectors = solution_vectors
        self.nqbit = []
        self.idx_start_data = self.get_indexes_data()
        self.encoded_reals = self.create_encoding()

    def create_encoding(self):
        """Create the enconding for all the unknowns


        Returns:
            list[RealEncoded]:
        """
        encoded_reals = []

        idx_vars = 0
        for sol_vec in self.solution_vectors:

            for i in range(sol_vec.size):
                var_base_name = sol_vec.base_name + "_%03d" % (idx_vars + 1)
                idx_vars += 1

                x = deepcopy(sol_vec.encoding)
                x.set_var_base_name(var_base_name)
                x.variables = x.create_variable()
                encoded_reals.append(x)
                self.nqbit.append(sol_vec.nqbit)

        return encoded_reals
