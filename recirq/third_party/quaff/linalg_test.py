import itertools

import cirq
import numpy as np
import pytest
from recirq.third_party import quaff as quaff


@pytest.mark.parametrize(
    "height, width, seed",
    [
        (height, width, quaff.random_seed())
        for height in range(1, 20)
        for width in range(1, 20)
        for _ in range(10)
    ],
)
def test_row_reduce(height, width, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, 2, (height, width))
    reduced, pivots = quaff.row_reduce(matrix)
    rank = len(pivots)
    assert not np.any(reduced[rank:])
    assert sorted(pivots) == list(pivots)
    for row, pivot_col in enumerate(pivots):
        assert not np.any(reduced[row, :pivot_col])
        assert np.array_equal(reduced[:, pivot_col], np.arange(height) == row)


@pytest.mark.parametrize(
    "height, width, seed",
    [
        (height, width, quaff.random_seed())
        for height in range(1, 10)
        for width in range(1, 10)
        for _ in range(10)
    ],
)
def test_get_coordinates(height, width, seed):
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, width)
    B = rng.integers(0, 2, (height, width))

    x, basis = quaff.get_coordinates(v, B, include_basis=True)

    actual_solutions = set()
    if x is not None:
        assert np.array_equal(quaff.dot(x, B), v)
        for r in range(height + 1):
            for rows in itertools.combinations(basis, r):
                solution = (x + np.sum(rows, axis=0, dtype=quaff.linalg.DTYPE)) % 2
                assert solution.shape == (height,)
                assert np.array_equal(quaff.dot(solution, B), v)
                actual_solutions.add(tuple(solution))

    expected_solutions = set()
    for y in itertools.product((0, 1), repeat=height):
        if np.array_equal(quaff.dot(y, B), v):
            expected_solutions.add(y)
    assert sorted(expected_solutions) == sorted(actual_solutions)


@pytest.mark.parametrize(
    "height, width, seed",
    [
        (height, width, quaff.random_seed())
        for height in range(1, 10)
        for width in range(1, 10)
        for _ in range(10)
    ],
)
def test_get_min_in_span(height, width, seed):
    rng = np.random.default_rng(seed)
    vector = rng.integers(0, 2, width)
    matrix = rng.integers(0, 2, (height, width))
    actual = quaff.get_min_in_span(vector, matrix)

    twos = 2 ** np.arange(width)

    def key(x):
        return np.dot(twos, (quaff.dot(x, matrix) + vector) % 2)

    solution = min(itertools.product((0, 1), repeat=height), key=key)
    expected = (quaff.dot(solution, matrix) + vector) % 2
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in range(1, 10) for _ in range(10)]
)
def test_get_inverse(n, seed):
    matrix = quaff.random_invertible_matrix(n, seed)
    inverse = quaff.get_inverse(matrix)
    eye = np.eye(n, dtype=quaff.linalg.DTYPE)
    assert np.array_equal(quaff.dot(matrix, inverse), eye)
    assert np.array_equal(quaff.dot(inverse, matrix), eye)


@pytest.mark.parametrize("n", range(1, 7))
def test_get_inverse_bad(n):
    matrix = np.eye(n, 2 * n, dtype=quaff.linalg.DTYPE)
    assert quaff.get_inverse(matrix) is None

    matrix = np.zeros((n, n), dtype=quaff.linalg.DTYPE)
    assert quaff.get_inverse(matrix) is None


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in range(1, 10) for _ in range(10)]
)
def test_get_lexicographic_basis(n, seed):
    matrix = quaff.random_invertible_matrix(n, seed)
    V = quaff.get_lexicographic_basis(matrix)
    for i in range(n):
        expected = quaff.get_min_in_span(matrix[i], matrix[i + 1 :])
        actual = V[i]
        assert np.array_equal(expected, actual)

        solution = quaff.get_coordinates(matrix[i], V[i:])
        assert solution[0]

    _, pivots = quaff.row_reduce(V)
    assert len(pivots) == n


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(10)]
)
def test_with_qubits_reversed(n, seed):
    rng = np.random.default_rng(seed)
    N = 2**n
    matrix = rng.random((N, N))
    new_matrix = quaff.with_qubits_reversed(matrix)
    assert new_matrix.shape == (N, N)
    for i in range(N):
        for j, x in enumerate(quaff.get_all_bitstrings(n)):
            new_j = quaff.bitstring_to_index(x[::-1])
            assert new_matrix[new_j, i] == matrix[j, i]


@pytest.mark.parametrize("n, seed", [(n, quaff.random_seed()) for n in range(1, 8)])
def test_apply_affine_transform(n, seed):
    rng = np.random.default_rng(seed)
    k = max(1, n // 2)
    N = 2**n
    bitstrings = rng.integers(0, 2, (k, n))
    transform = quaff.random_invertible_matrix(n, rng)
    amplitudes = 2 * (rng.random(k) + 1j * rng.random(k))
    assert amplitudes.shape == (k,)
    indices = quaff.bitstrings_to_indices(bitstrings)
    assert indices.shape == (k,)
    state = sum(
        a * cirq.one_hot(index=i, shape=N, dtype=complex)
        for a, i in zip(amplitudes, indices)
    )
    assert state.shape == (N,)
    offset = rng.integers(0, 2, n).astype(quaff.linalg.DTYPE)
    for b in (None, offset):
        new_bitstrings = quaff.apply_affine_transform_to_bitstrings(
            transform, bitstrings, offset=b
        )
        assert new_bitstrings.shape == bitstrings.shape
        new_indices = quaff.bitstrings_to_indices(new_bitstrings)
        assert new_indices.shape == indices.shape
        expected = sum(
            a * cirq.one_hot(index=i, shape=N, dtype=complex)
            for a, i in zip(amplitudes, new_indices)
        )
        assert expected.shape == (N,)
        actual = quaff.apply_affine_transform_to_state_vector(
            transform, state, offset=b
        )
        assert np.allclose(expected, actual)

    state = quaff.random_state(n, rng)

    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit(cirq.X(q) for i, q in enumerate(qubits) if offset[i])
    unitary = circuit.unitary(qubit_order=qubits)
    expected = np.dot(unitary, state)

    transform = np.eye(n, dtype=quaff.linalg.DTYPE)
    actual = quaff.apply_affine_transform_to_state_vector(
        transform, state, offset=offset
    )
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "n, seed", ((quaff.RNG.integers(1, 10), quaff.random_seed()) for _ in range(10))
)
def test_invert_permutation(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv_perm = quaff.invert_permutation(perm)
    assert np.array_equal(perm[inv_perm], np.arange(n))
    assert np.array_equal(inv_perm[perm], np.arange(n))
