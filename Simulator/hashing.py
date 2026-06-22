import numpy as np
from sympy import nextprime
from scipy.constants import *
from scipy.linalg import toeplitz


class Hashing:
    def create_hash(self, seed, x, length):
        raise NotImplementedError("Subclasses must implement this method")


class CarterWegmanHash(Hashing):
    def create_hash(self, seed, x, length):
        # https://en.wikipedia.org/wiki/Universal_hashing
        prime = nextprime(2**length)
        a = seed[:len(seed)//2]
        b = seed[len(seed)//2:]
        a_int = int(a, 2)  # Half the seed
        b_int = int(b, 2)  # other half of the seed
        x_int = int(x, 2)  # The input data
        bit_len = (2 ** length) - 1  # Calculate the length in bytes
        h = ((a_int * x_int + b_int) % prime) % bit_len
        return h


class ToeplitzHash(Hashing):
    # https://en.wikipedia.org/wiki/Toeplitz_Hash_Algorithm
    def _toeplitz_matrix(self, out_len, in_len, seed):
        row = np.array([int(i) for i in seed[:out_len]])
        col = np.array([int(i) for i in seed[out_len:out_len+in_len]])
        toep_mat = toeplitz(row, col)
        return toep_mat

    def create_hash(self, seed, x, length):
        matrix = self._toeplitz_matrix(length, len(x), seed)
        # generate a matrix with the seed and x
        k = np.array([int(i) for i in (x)])
        h = np.dot(matrix, k) % 2
        # convert to one big integer
        h_int = int(''.join(str(int(i)) for i in h), 2)
        return h_int
