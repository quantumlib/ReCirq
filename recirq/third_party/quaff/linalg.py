from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
from quaff import indexing

DTYPE = np.uint8


def dot(*args, **kwargs):
    """Dot product over 𝔽₂. Same syntax as numpy.dot."""
    return np.dot(*args, **kwargs) % 2


def row_reduce(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Produces a matrix in reduced row echelon form, with corresponding pivots.

    Args:
        matrix: The matrix to row reduce.

    Returns:
        Tuple of matrix in reduced row echelon form and the corresponding pivots.

    All arithmetic is over 𝔽₂.
    """
    matrix = np.copy(matrix) % 2
    n_rows, n_cols = matrix.shape
    pivot_row = 0
    pivot_col = 0
    pivots = []
    while (pivot_row < n_rows) and (pivot_col < n_cols):
        row = pivot_row + np.argmax(matrix[pivot_row:, pivot_col])
        if matrix[row, pivot_col]:
            matrix[[pivot_row, row]] = matrix[[row, pivot_row]]
            for row in range(pivot_row + 1, n_rows):
                if matrix[row, pivot_col]:
                    matrix[row] = (matrix[row] + matrix[pivot_row]) % 2
            pivots.append((pivot_row, pivot_col))
            pivot_row += 1
        pivot_col += 1
    for pivot_row, pivot_col in reversed(pivots):
        for row in range(pivot_row):
            if matrix[row, pivot_col]:
                matrix[row] = (matrix[row] + matrix[pivot_row]) % 2
    return matrix, tuple(col for _, col in pivots)


def get_inverse(
    matrix: np.ndarray, raise_error_if_not_invertible: bool = False
) -> Optional[np.ndarray]:
    """Returns the inverse of a matrix over 𝔽₂.

    Args:
        matrix: The matrix to invert.

    Returns: The inverse if it exists, else None.
    """
    n = len(matrix)
    if matrix.shape != (n, n):
        return None
    eye = np.eye(n, dtype=DTYPE)
    system = np.hstack((matrix, eye))
    reduced, _ = row_reduce(system)
    if np.array_equal(reduced[:, :n], eye):
        return reduced[:, n:]
    if raise_error_if_not_invertible:
        raise RuntimeError("Matrix is not invertible.")
    return None


def get_coordinates(
    vector: np.ndarray, basis: np.ndarray, include_basis: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Given vector v and matrix B, finds x such that v = Bᵀ·x if one exists.

    Args:
        vector: The vector v.
        basis: The matrix B.
        include_basis: Optional. Basis of solutions.

    Returns:
        Vector x such that v = Bᵀ·x if one exists; None otherwise. If
        include_basis is true, returns (x, Y) such that x ⊕ span{rows of Y} is
        the solution space.

    """
    m, n = basis.shape
    system = np.append(basis, vector[np.newaxis, :], axis=0).T
    assert system.shape == (n, m + 1)
    reduced, pivots = row_reduce(system)
    if pivots and (pivots[-1] == m):
        return (None, np.zeros((0, m), dtype=DTYPE)) if include_basis else None

    solution = np.zeros(m, dtype=DTYPE)
    for r, c in enumerate(pivots):
        solution[c] = reduced[r, m]
    if include_basis:
        free_columns = sorted(set(range(m)) - set(pivots))
        solution_rank = len(free_columns)
        solution_basis = np.tensordot(
            np.ones(solution_rank, dtype=DTYPE), solution, axes=0
        )
        if not solution_rank:
            return solution, solution_basis
        solution_basis = np.zeros((solution_rank, m), dtype=DTYPE)
        assert max(free_columns) < m
        solution_basis[:, free_columns] = np.eye(solution_rank, dtype=DTYPE)
        pivots = list(pivots)
        for i, f in enumerate(free_columns):
            for r, c in enumerate(pivots):
                solution_basis[i, c] += reduced[r, f]
        solution_basis %= 2
        return solution, solution_basis
    return solution


def is_invertible(matrix: np.ndarray):
    n = len(matrix)
    if matrix.shape != (n, n):
        return False
    _, pivots = row_reduce(matrix)
    return len(pivots) == n


def is_tril_and_unit_diag(matrix: np.ndarray) -> bool:
    """Determines if a matrix is lower-triangular and unit-diagonal."""
    return (np.diag(matrix) == 1).all() and not np.triu(matrix, 1).any()


def nw_tri(matrix: np.ndarray) -> np.ndarray:
    """Returns the northwest triangular part of the given matrix."""
    return np.flip(np.tril(np.flip(matrix, 0), 0), 0)


def get_min_in_span(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    _, n_cols = matrix.shape
    if not n_cols:
        return np.zeros(0, dtype=DTYPE)
    if not np.any(matrix):
        return vector

    solution = get_coordinates(vector, matrix)
    if solution is not None:
        return np.zeros(n_cols, dtype=DTYPE)

    for j in range(n_cols):
        ansatz = np.eye(1, n_cols - j, dtype=DTYPE).flatten()
        suffix = (ansatz + vector[j:]) % 2
        solution, solution_basis = get_coordinates(
            suffix, matrix[:, j:], include_basis=True
        )
        if solution is not None:
            new_vector = (dot(solution, matrix) + vector) % 2
            if not len(solution_basis):
                return new_vector
            new_basis = np.array([dot(b, matrix) for b in solution_basis])
            return np.concatenate(
                (get_min_in_span(new_vector[:j], new_basis[:, :j]), ansatz)
            )


def get_lexicographic_basis(matrix: np.ndarray) -> np.ndarray:
    """Returns a lexicographically sorted basis (over 𝔽₂).

    For input matrix with rows (a₁, …, aₙ), returns a matrix with rows (v₁, …,
    vₙ) such that vᵢ= min{aᵢ⊕ span{aⱼ: i < j <=n }} for 1 < i < n, where x < v
    iff there exists k such that xₖ = 0, yₖ = 1, and for all j > k, xⱼ= yⱼ.
    """
    V = np.empty_like(matrix)
    for i, row in enumerate(matrix):
        V[i] = get_min_in_span(row, matrix[i + 1 :])
    return V


def is_nw_tri(matrix: np.ndarray) -> bool:
    """Determines if a matrix is northwest triangular."""
    return np.array_equiv(np.triu(np.flip(matrix, axis=0), 1), 0)


def with_qubits_reversed(
    operator: np.ndarray, num_qubits: Optional[int] = None
) -> np.ndarray:
    """Appends a qubit reversal to the given operator.

    Args:
        operator: An 2ⁿ×2ⁿ matrix.
        num_qubits: The number n of qubits. Defaults to None, in which case it
            is inferred from the shape of operator.

    Returns: An M×2ⁿ operator equal to applying the given operator and
        reversing the qubits.corresponding to reversing the qubit ordering of
        the given matrix.
    """
    M, N = operator.shape
    if num_qubits is None:
        num_qubits = indexing.log2(N)
    axes = list(reversed(range(num_qubits))) + [num_qubits]
    return np.transpose(operator.reshape((2,) * num_qubits + (M,)), axes=axes).reshape(
        (N, M)
    )


def apply_affine_transform_to_bitstrings(
    matrix: np.ndarray, bitstrings: np.ndarray, *, offset: Optional[np.ndarray] = None
) -> np.ndarray:
    """Applies an affine transform Ax + b to bitstrings (over 𝔽₂).

    Args:
        matrix: The n×n matrix A.
        bitstrings: The bitstrings to transform.
        offset: The n-dimensional vector b. Defaults to all zeros.

    Returns: The transformed bitstrings.
    """
    transformed_bitstrings = np.tensordot(bitstrings, matrix, axes=(-1, -1))
    if offset is not None:
        transformed_bitstrings += offset
    return transformed_bitstrings % 2


def apply_affine_transform_to_state_vector(
    matrix: np.ndarray, state_vector: np.ndarray, *, offset: Optional[np.ndarray] = None
) -> np.ndarray:
    """Applies an affine transform Ax + b in the computational basis to a state vector.

    Args:
        matrix: The n×n matrix A.
        state_vector: The 2ⁿ-dimensional state vector.
        offset: The n-dimensional vector b. Defaults to all zeros.

    Returns: The transformed state vector.

    The matrix A must be nonsingular.
    """

    n = len(matrix)
    N = 2**n
    bitstrings = indexing.get_all_bitstrings(n)
    assert bitstrings.shape == (N, n)
    new_bitstrings = apply_affine_transform_to_bitstrings(
        matrix, bitstrings, offset=offset
    )
    assert new_bitstrings.shape == (N, n)
    new_indices = indexing.bitstrings_to_indices(new_bitstrings)
    assert new_indices.shape == (N,)
    old_to_new = {j: i for i, j in enumerate(new_indices)}
    new_to_old = np.array([old_to_new[i] for i in range(N)])
    assert new_to_old.shape == (N,)
    return state_vector[new_to_old]


def invert_permutation(perm: Iterable[int]) -> Optional[np.ndarray]:
    perm = {j: i for i, j in enumerate(perm)}
    return np.array(tuple(perm[i] for i in range(len(perm))))


def vector_to_tight_str(x: np.ndarray) -> str:
    return "".join(str(xx) for xx in x)


def matrix_to_tight_str(X: np.ndarray) -> str:
    return "|".join(vector_to_tight_str(x) for x in X)


def tuplify(array: np.ndarray, wrap: Callable = DTYPE):
    if array.ndim > 1:
        return tuple(tuplify(x, wrap) for x in array)
    return tuple(wrap(x) for x in array)


def tuple_of_tuples(obj: Any, wrap: Callable = DTYPE) -> Tuple[Tuple[Any, ...], ...]:
    return tuple(tuple(wrap(x) for x in row) for row in obj)


BooleanVector = Tuple[bool, ...]
BooleanMatrix = Tuple[BooleanVector, ...]
