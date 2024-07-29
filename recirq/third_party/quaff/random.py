import itertools
from typing import Optional, Tuple

import numpy as np
from recirq.third_party.quaff import indexing, linalg, sampling

SEED = 2349230498
RNG = np.random.default_rng(SEED)


def random_seed(size=None):
    return RNG.integers(0, 2**63, size=size)


def random_state(num_qubits, rng):
    N = 2**num_qubits
    state = rng.standard_normal(N).astype(complex)
    state += 1j * rng.standard_normal(N)
    state /= np.linalg.norm(state)
    return state


def random_matrix(
    height: int, width: Optional[int] = None, *, seed=None, high=2, dtype=linalg.DTYPE
) -> np.ndarray:
    """Produces a random square 0-1 matrix."""
    if width is None:
        width = height
    rng = np.random.default_rng(seed)
    return rng.integers(0, high, (height, width), dtype=dtype)


def random_symmetric_matrix(
    n: int, *, seed=None, high=2, dtype=linalg.DTYPE
) -> np.ndarray:
    matrix = random_matrix(n, seed=seed, high=high, dtype=dtype)
    for i, j in itertools.combinations(range(n), 2):
        matrix[j, i] = matrix[i, j]
    return matrix


def random_tril_and_unit_diag_matrix(
    n: int, *, seed=None, high=2, dtype=linalg.DTYPE
) -> np.ndarray:
    matrix = random_matrix(n, seed=seed, high=high, dtype=dtype)
    matrix[np.diag_indices(n)] = 1
    matrix[indexing.triu_indices(n)] = 0
    return matrix


def random_vector(n: int, seed=RNG):
    return np.random.default_rng(seed).integers(0, 2, n, dtype=linalg.DTYPE)


def random_invertible_matrix(n: int, seed=None) -> np.ndarray:
    """Uniformly samples a nonsingular matrix over 𝔽₂.
    Args:
        n: The size of the matrix.
        seed: The randomness seed.

    Returns: A uniformly random non-singula matrix.
    """
    return np.array(
        sampling.InvertibleMatrixSampler(n).sample(np.random.default_rng(seed))
    )


def random_nw_tri_matrix(n: int, seed=None) -> np.ndarray:
    """Produces a random square 0-1 matrix that is northwest triangular."""
    return linalg.nw_tri(random_matrix(n, seed=seed))


def random_invertible_nw_tri_matrix(n: int, seed=None) -> np.ndarray:
    """Produces a random square 0-1 matrix that is invertible and northwest triangular."""
    matrix = np.eye(n, dtype=linalg.DTYPE)
    rng = np.random.default_rng(seed)
    for i in range(n):
        matrix[i, i + 1 :] = rng.integers(0, 2, n - i - 1)
    return np.flip(matrix, 1)


def random_permutation(n: int, rng: Optional[np.random.Generator] = None) -> Tuple[int]:
    """Generates a random permutation according to the Mallows distribution.

    Args:
        n: The number of elements.
        rng: Optionally, The RNG.

    Returns: The permutation S(i) = j as a 1D array [S(1), S(2), ...]

    Algorithm 3 from arXiv:2003.09412.
    """
    rng = np.random.default_rng(rng)
    return sampling.MallowsSampler(n).sample(rng)
