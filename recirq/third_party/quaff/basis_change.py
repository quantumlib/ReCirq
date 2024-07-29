from typing import Iterable, List, Sequence, Tuple

import cirq
import numpy as np
from quaff import indexing, linalg, random


class BasisChangeGate(cirq.Gate):
    def __init__(self, matrix: np.ndarray):
        """A gate that applies a basis change (over 𝔽₂) to computational basis states.

        Args:
            matrix: The basis change.
        """
        self.matrix = np.copy(matrix)

    def num_qubits(self):
        return len(self.matrix)

    def _unitary_(self):
        N = 2 ** self.num_qubits()
        bitstrings = indexing.get_all_bitstrings(self.num_qubits())
        LHS = np.tensordot(
            bitstrings, np.ones(N, dtype=linalg.DTYPE), axes=0
        ).transpose((0, 2, 1))
        RHS = np.tensordot(
            (bitstrings @ self.matrix.T) % 2, np.ones(N, dtype=linalg.DTYPE), axes=0
        ).transpose((2, 0, 1))
        return (LHS == RHS).all(axis=-1)

    @classmethod
    def random(cls, num_qubits, seed=None):
        rng = np.random.default_rng(seed)
        matrix = random.random_invertible_matrix(num_qubits, seed=rng)
        return cls(matrix)

    def __pow__(self, exponent) -> "BasisChangeGate":
        if not np.issubdtype(type(exponent), np.integer):
            raise ValueError("Exponent must be an integer.")
        if exponent < 0:
            return self.inverse() ** (-exponent)
        return type(self)(np.linalg.matrix_power(self.matrix, exponent) % 2)

    def inverse(self) -> "BasisChangeGate":
        return type(self)(linalg.get_inverse(self.matrix))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError
        return np.array_equal(self.matrix % 2, other.matrix % 2)

    def __str__(self):
        return "BasisChangeGate([{}])".format(
            ",".join(
                "[{}]".format(",".join(str(e) for e in row)) for row in self.matrix
            )
        )

    @classmethod
    def from_unitary(cls, unitary: np.ndarray):
        if not np.isreal(unitary).all():
            raise ValueError("Unitary not real.")
        unitary = np.real(unitary)
        N = len(unitary)
        n = indexing.log2(N)
        bitstrings = indexing.get_all_bitstrings(n)
        masks = (1 << np.arange(n))[::-1]
        matrix = np.zeros((n, n), dtype=linalg.DTYPE)
        for i, mask in enumerate(masks):
            matrix[:, i] = unitary[:, mask] @ bitstrings
        return cls(matrix)

    def expand(self, new_num_qubits: int, indices: np.ndarray) -> "BasisChangeGate":
        indices = np.array(tuple(indices))
        matrix = np.zeros((new_num_qubits,) * 2, dtype=linalg.DTYPE)
        matrix[np.ix_(indices, indices)] = self.matrix
        new_indices = np.array(
            sorted(set(range(new_num_qubits)) - set(indices)), dtype=linalg.DTYPE
        )
        num_new_indices = new_num_qubits - self.num_qubits()
        matrix[np.ix_(new_indices, new_indices)] = np.eye(num_new_indices)
        return type(self)(matrix)

    def copy(self):
        return type(self)(np.copy(self.matrix))

    @classmethod
    def from_gate(cls, gate: cirq.Gate):
        if gate == cirq.CNOT:
            return cls([[1, 0], [1, 1]])
        raise NotImplementedError

    def __matmul__(self, other: "BasisChangeGate") -> "BasisChangeGate":
        if not isinstance(other, type(self)):
            raise TypeError("Can only matmul by an instance of BasisChangeGate")
        if self.num_qubits() != other.num_qubits():
            raise ValueError(
                "Can only matmal basis change gates with the same number of qubits."
            )
        return type(self)((self.matrix @ other.matrix) % 2)

    def _decompose_(self, qubits):
        if self.num_qubits() <= 2:
            return None
        clearing_network, nw_tri_matrix = get_clearing_network(self.matrix, qubits)
        reversal_network = list(get_reversal_network(nw_tri_matrix, qubits))
        yield cirq.inverse(reversal_network)
        yield cirq.inverse(clearing_network)

    @classmethod
    def identity(cls, num_qubits: int) -> "BasisChangeGate":
        return cls(np.eye(num_qubits, dtype=linalg.DTYPE))


def get_clearing_network(
    matrix: np.ndarray, qubits: Sequence[cirq.Qid], include_output_matrix: bool = False
) -> Tuple[List[cirq.Operation], np.ndarray]:
    """Returns a clearing network as iterable of basis change gates.

    Args:
        matrix: The matrix to make NW triangular. Must be nonsingular.

    Returns: Iterable of `BasisChangeGate`s.

    The overall effect of the network is to transform a matrix to a NW matrix.

    See "Computation at a Distance" by Kutin et al.
    """
    n = len(matrix)
    values = np.copy(matrix)
    V = linalg.get_lexicographic_basis(matrix)
    pi = tuple(np.argmax(np.flip(V, 1), 1))
    W = np.empty_like(matrix)
    for i in range(n):
        W[pi[i]] = V[i]
    assert linalg.is_nw_tri(W)

    labels = np.array(pi)

    ops = []

    for layer in range(n):
        for i in range(layer % 2, n - 1, 2):
            j, k = labels[i : i + 2]
            if j < k:
                continue
            labels[i : i + 2] = k, j
            u, v = values[i : i + 2]
            W_minor = np.vstack((W[:k], W[k + 1 :]))
            for a, b in [(0, 1), (1, 1), (1, 0)]:
                rhs = (a * u + b * v) % 2
                solution = linalg.get_coordinates(rhs, W_minor)
                if solution is not None:
                    assert np.array_equal(linalg.dot(solution, W_minor), rhs)
                    break
            solution = np.insert(solution, k, 0)
            assert np.array_equal(linalg.dot(solution, W), rhs)

            if (a, b) == (1, 0):
                values[i] = (u + v) % 2
            values[i + 1] = rhs

            op = BasisChangeGate(np.array([[1, a * (1 - b)], [a, b]]))(
                *qubits[i : i + 2]
            )
            ops.append(op)

    assert np.array_equal(labels, np.arange(n))
    assert linalg.is_nw_tri(values)
    return ops, values


def get_reversal_network(
    matrix: np.ndarray, qubits: Sequence[cirq.Qid]
) -> Iterable[cirq.Operation]:
    """Returns a reversal network as iterable of basis change gates.

    Args:
        matrix: The matrix to transform to identity. Must be NW triangular and nonsingular.

    Returns: Iterable of `BasisChangeGate`s.

    The overall effect of the network is to transform a NW triangular matrix to the identity.

    See "Computation at a Distance" by Kutin et al.
    """
    n = len(matrix)
    labels = np.arange(n)[::-1]
    values = np.copy(matrix)
    for layer in range(n):
        for i in range(layer % 2, n - 1, 2):
            gate_matrix = np.array([[0, 1], [1, 0]], dtype=linalg.DTYPE)
            if values[i][labels[i + 1]]:
                values[i] = (values[i] + values[i + 1]) % 2
                gate_matrix[1, 1] = 1
            values[i : i + 2] = values[i : i + 2][::-1]
            labels[i : i + 2] = labels[i : i + 2][::-1]

            yield BasisChangeGate(gate_matrix)(*qubits[i : i + 2])
