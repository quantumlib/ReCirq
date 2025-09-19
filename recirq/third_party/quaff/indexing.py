import itertools
from typing import Tuple

import numpy as np


def get_all_bitstrings(n: int, big_endian: bool = True, dtype=bool) -> np.ndarray:
    """Returns all bitstrings of length n in lexicographic order.

    Args:
        n: The length of the bitstrings.
        big_endian: Whether or not ordering is big-endian. Defaults to True.
        dtype: The dtype of the bitstrings.

    Returns: The bitstrings as an array.
    """
    integers = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)
    if big_endian:
        masks = masks[::-1]
    return (integers[:, np.newaxis] & masks).astype(bool).astype(dtype)


def get_unit_vector_index(n: int, i: int) -> int:
    """The integer encoded (via big-endian order) by a one-hot bitstring.

    Args:
        n: The length of the bitstring.
        i: The location of the sole one.

    Returns: The encoded integer.
    """
    return 1 << (n - i - 1)


def get_unit_vector_indices(n: int) -> np.ndarray:
    """The integers encoded (via big-endian order) by all one-hot bitstrings of
    the given length."""
    return (1 << np.arange(n))[::-1]


def bitstring_to_index(x: np.ndarray) -> np.ndarray:
    """Returns the integer encoded (via big-endian order) of the given
    bitstring."""
    x = np.array(x)
    (n,) = x.shape
    return np.dot(1 << (n - np.arange(n) - 1), x)


def bitstrings_to_indices(X: np.ndarray) -> np.ndarray:
    """Returns the integers encoded (via big-endian order) of the given
    bitstrings."""
    X = np.array(X)
    _, n = X.shape
    return (X @ (1 << (n - np.arange(n) - 1))[:, np.newaxis])[:, 0]


def index_to_bitstring(n: int, i: int) -> np.ndarray:
    """Returns the bitstring encoding (in big-endian order) the given integer.

    Args:
        n: The length of the bitstring.
        i: The encoded integer.

    Returns: The bitstring.
    """
    return (i & get_unit_vector_indices(n)).astype(bool)


def offdiag_indices(n: int) -> np.ndarray:
    """Return the indices to access the nondiagonal entries of an array.
    Complements numpy.diag_indices."""
    return np.nonzero(np.ones((n, n)) - np.eye(n))


def tril_indices(n: int) -> Tuple[np.ndarray, ...]:
    """Return the indices to access the entries of an array below the diagonal."""
    if n == 1:
        return (np.zeros(0, dtype=int),) * 2
    return tuple(np.array(tuple(itertools.combinations(reversed(range(n)), 2))).T)


def triu_indices(n: int) -> Tuple[np.ndarray, ...]:
    """Return the indices to access the entries of an array above the diagonal."""
    return tril_indices(n)[::-1]


def log2(N: int) -> int:
    """Returns the base-2 logarithm of the given integer. Raises an error if
    the input is not a power of 2."""
    n = int(np.around(np.log2(N)))
    if N != 2**n:
        raise ValueError(f"{N} is not a power of 2.")
    return n
