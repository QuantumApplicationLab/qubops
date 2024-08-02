from sympy.matrices import Matrix
import numpy as np
from copy import deepcopy
from .encodings import RangedEfficientEncoding


class SolutionVector(object):

    def __init__(self, size, nqbit, encoding, range=1.0, offset=0.0, base_name="x"):
        """Encode the solution vector in a list of RealEncoded

        Args:
            size (int): number of unknonws in the vector (i.e. size of the system)
            nqbit (int): number of qbit required per unkown
            base_name (str, optional): base name of the unknowns Defaults to 'x'.
            only_positive (bool, optional):  Defaults to False.
        """
        self.size = size
        self.nqbit = nqbit
        self.base_name = base_name
        self.encoding = encoding
        self.range = range
        self.offset = offset

        if not isinstance(self.range, list):
            self.range = [self.range] * size

        if not isinstance(self.offset, list):
            self.offset = [self.offset] * size

        self.encoded_reals = self.create_encoding()

    def create_encoding(self):
        """Create the eocnding for all the unknowns


        Returns:
            list[RealEncoded]:
        """
        encoded_reals = []
        is_ranged_encoding = self.encoding == RangedEfficientEncoding
        for i in range(self.size):
            var_base_name = self.base_name + "_%03d" % (i + 1)
            if is_ranged_encoding:
                args = (self.nqbit, self.range[i], self.offset[i], var_base_name)
            else:
                args = (self.nqbit, var_base_name)
            encoded_reals.append(self.encoding(*args))
        return encoded_reals

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

        sol = []
        for i, real in enumerate(self.encoded_reals):
            local_data = data[i * self.nqbit : (i + 1) * self.nqbit]
            sol.append(real.decode_polynom(local_data))
        return np.array(sol) + np.array(self.offset)


class SolutionVector_V2(SolutionVector):

    def __init__(self, size, encoding, base_name="x"):
        """_summary_

        Args:
            size (_type_): _description_
            encoding (_type_): _description_
            base_name (str)

        Returns:
            _type_: _description_
        """

        self.size = size
        self.base_name = base_name
        self.encoding = encoding
        self.nqbit = encoding.nqbit
        self.offset = 0.0
        self.encoded_reals = self.create_encoding()

    def create_encoding(self):
        """Create the eocnding for all the unknowns


        Returns:
            list[RealEncoded]:
        """
        encoded_reals = []

        for i in range(self.size):
            var_base_name = self.base_name + "_%03d" % (i + 1)
            x = deepcopy(self.encoding)
            x.set_var_base_name(var_base_name)
            x.variables = x.create_variable()
            encoded_reals.append(x)
        return encoded_reals
