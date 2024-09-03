from sympy import Symbol
import itertools
import numpy as np


class BaseQbitEncoding(object):

    def __init__(self, nqbit, var_base_name):
        """Encode a  single real number in a

        Args:
            nqbit (int): number of qbit required in the expansion
            var_base_name (str): base names of the different qbits
            only_positive (bool, optional): Defaults to False.
        """
        self.nqbit = nqbit
        self.var_base_name = var_base_name
        self.variables = self.create_variable()

    def set_var_base_name(self, var_base_name):
        """set the variable base name

        Args:
            var_base_name (_type_): _description_
        """
        self.var_base_name = var_base_name

    def create_variable(self):
        """Create all the variabes/qbits required for the expansion

        Returns:
            list: list of Symbol
        """
        variables = []
        for i in range(self.nqbit):
            variables.append(Symbol(self.var_base_name + "_%03d" % (i + 1)))
        return variables

    def create_polynom(self):
        raise NotImplementedError("Implement create_polynom")

    def decode_polynom(self, data):
        raise NotImplementedError("Implement decode_polynom")

    def get_max_value(self):
        """Get the maximum value of the encoding

        Returns:
            float: max value
        """
        return self.decode_polynom([1] * self.nqbit)

    def get_possible_values(self):
        """get all the posible values encoded

        Returns:
            _type_: _description_
        """

        values = []
        for data in itertools.product([0, 1], repeat=self.nqbit):
            values.append(self.decode_polynom(list(data)[::-1]))
        return values

    def find_closest(self, float):
        """finds the closest possible encoded number to float

        Args:
            float (_type_): _description_
        """

        min_diff = 1e12
        closest_value = None
        binary_encoding = None
        for data in itertools.product([0, 1], repeat=self.nqbit):
            val = self.decode_polynom(list(data)[::-1])
            if np.abs(val - float) < min_diff:
                min_diff = np.abs(val - float)
                closest_value = val
                binary_encoding = list(data)[::-1]

        return closest_value, binary_encoding

    def get_average_precision(self):
        """get the mean precision on the encoded variables"""
        vals = self.get_possible_values()
        z = vals - np.roll(vals, 1)
        return np.mean(z[1:])


class RangedEfficientEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, range, offset, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0
        self.int_max = 2 ** (nqbit - 1) - 1

        if not isinstance(range, list):
            range = [range] * nqbit
        self.max_absval = [r / self.int_max for r in range]

        self.offset = offset

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = (
            self.offset
            - (2 ** (self.nqbit - 1)) * self.variables[0] * self.max_absval[0]
        )
        for i in range(self.nqbit - 1):
            out += 2 ** (i) * self.variables[i + 1] * self.max_absval[i]
        return out

    def decode_polynom(self, data):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = self.offset - (2 ** (self.nqbit - 1)) * data[0] * self.max_absval[0]
        for i in range(self.nqbit - 1):
            out += 2 ** (i) * data[i + 1] * self.max_absval[i]
        return out


class EfficientEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0
        self.int_max = 2 ** (nqbit - 1) - 1

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = -(2 ** (self.nqbit - 1)) * self.variables[0]
        for i in range(self.nqbit - 1):
            out += 2 ** (i) * self.variables[i + 1]
        return out / self.int_max

    def decode_polynom(self, data):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = -(2 ** (self.nqbit - 1)) * data[0]
        for i in range(self.nqbit - 1):
            out += 2 ** (i) * data[i + 1]
        return out / self.int_max


class RealQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0
        for i in range(self.nqbit // 2):
            out += 2 ** (i - self.base_exponent) * self.variables[i]
            out -= 2 ** (i - self.base_exponent) * self.variables[self.nqbit // 2 + i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        for i in range(self.nqbit // 2):
            out += 2 ** (i - self.base_exponent) * data[i]
            out -= 2 ** (i - self.base_exponent) * data[self.nqbit // 2 + i]
        return out


class RealUnitQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0
        self.int_max = None
        assert (nqbit - 1) % 2 == 0

    def find_int_max(self):
        """Find the amx value of the encoding"""
        i = 0
        self.int_max = 2 ** (i - self.base_exponent)

        for i in range(1, (self.nqbit - 1) // 2):
            self.int_max += 2 ** (i - self.base_exponent)

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0

        self.find_int_max()

        i = 0
        out += 2 ** (i - self.base_exponent) / self.int_max * self.variables[i]

        for i in range(1, (self.nqbit - 1) // 2 + 1):

            out += 2 ** (i - self.base_exponent) / self.int_max * self.variables[i]
            out -= (
                2 ** (i - self.base_exponent)
                / self.int_max
                * self.variables[(self.nqbit - 1) // 2 + i]
            )
        return out

    def decode_polynom(self, data):
        out = 0.0

        if self.int_max is None:
            self.find_int_max()

        i = 0
        out += 2 ** (i - self.base_exponent) / self.int_max * data[i]

        for i in range(1, (self.nqbit - 1) // 2 + 1):
            out += 2 ** (i - self.base_exponent) / self.int_max * data[i]
            out -= (
                2 ** (i - self.base_exponent)
                / self.int_max
                * data[(self.nqbit - 1) // 2 + i]
            )
        return out


class PositiveQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name, offset=0, step=1):
        super().__init__(nqbit, var_base_name)
        self.offset = offset
        self.step = step

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = self.offset
        for i in range(self.nqbit):
            out += self.step * 2**i * self.variables[i]
        return out

    def decode_polynom(self, data):
        out = self.offset
        for i in range(self.nqbit):
            out += self.step * 2**i * data[i]
        return out


class DiscreteValuesEncoding(BaseQbitEncoding):

    def __init__(self, values, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.discrete_values = values
        self.coefs = self.get_coefficients()
        self.offset = 0

    def get_coefficients(self):
        """get the lstqst coefficients"""
        nvalues = len(self.discrete_values)
        nqbit = self.nqbit
        A = np.zeros((nvalues, nqbit + 1))
        c = [1] + [2**i for i in range(nqbit)]
        for idx in range(nvalues):
            row = [1] + [float(i) for i in np.binary_repr(idx, width=nqbit)][::-1]
            A[idx, :] = row
        A = A * c

        coefs, res, rank, s = np.linalg.lstsq(A, self.discrete_values)

        return coefs

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = self.coefs[0]
        for i in range(self.nqbit):
            out += self.coefs[i + 1] * 2**i * self.variables[i]
        return out

    def decode_polynom(self, data):
        out = self.coefs[0]
        for i in range(self.nqbit):
            out += self.coefs[i + 1] * 2**i * data[i]
        return out
