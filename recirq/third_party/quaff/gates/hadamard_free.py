import dataclasses
from dataclasses import dataclass
from typing import Any, Tuple

import cirq
import numpy as np
from recirq.third_party.quaff import indexing, linalg, random, testing


@dataclass
class HadamardFreeGate(cirq.Gate):
    num_qubits: dataclasses.InitVar[int]
    _num_qubits: int = dataclasses.field(init=False)
    parity_matrix: np.ndarray = None
    phase_matrix: np.ndarray = None
    parity_shift: np.ndarray = None
    phase_shift: np.ndarray = None
    phase_constant: float = 0

    def __post_init__(self, num_qubits):
        self._num_qubits = num_qubits
        if self.parity_matrix is None:
            self.parity_matrix = np.eye(num_qubits, dtype=linalg.DTYPE)
        else:
            self.parity_matrix = np.array(self.parity_matrix, dtype=linalg.DTYPE)
        if self.phase_matrix is None:
            self.phase_matrix = np.zeros((num_qubits,) * 2, dtype=linalg.DTYPE)
        else:
            self.phase_matrix = np.array(self.phase_matrix, dtype=linalg.DTYPE)
        if self.parity_shift is None:
            self.parity_shift = np.zeros(num_qubits, dtype=linalg.DTYPE)
        else:
            self.parity_shift = np.array(self.parity_shift, dtype=linalg.DTYPE)
        if self.phase_shift is None:
            self.phase_shift = np.zeros(num_qubits, dtype=linalg.DTYPE)
        else:
            self.phase_shift = np.array(self.phase_shift, dtype=linalg.DTYPE)

    @classmethod
    def _from_json_dict_(cls, *, _num_qubits, **kwargs):
        del kwargs["cirq_type"]
        return cls(num_qubits=_num_qubits, **kwargs)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return "quaff"

    def _num_qubits_(self):
        return self._num_qubits

    def _unitary_(self):
        return self.unitary()

    def __bool__(self):
        if self.phase_matrix.any():
            return True
        if self.phase_shift.any():
            return True
        if self.parity_shift.any():
            return True
        if not np.array_equal(
            self.parity_matrix, np.eye(self._num_qubits, dtype=linalg.DTYPE)
        ):
            return True
        if not np.isclose(self.phase_constant, 0):
            return True
        return False

    def copy(self):
        return type(self)(
            num_qubits=self._num_qubits,
            parity_matrix=np.copy(self.parity_matrix),
            phase_matrix=np.copy(self.phase_matrix),
            parity_shift=np.copy(self.parity_shift),
            phase_shift=np.copy(self.phase_shift),
            phase_constant=self.phase_constant,
        )

    def validate(self) -> bool:
        n = self._num_qubits
        if not testing.is_boolean_array_of_shape(self.parity_matrix, (n, n)):
            return False
        if not linalg.is_invertible(self.parity_matrix):
            return False
        if not testing.is_boolean_array_of_shape(self.phase_matrix, (n, n)):
            return False
        if not np.array_equal(self.phase_matrix, self.phase_matrix.T):
            return False
        if not testing.is_boolean_array_of_shape(self.parity_shift, (n,)):
            return False
        if not testing.is_boolean_array_of_shape(self.phase_shift, (n,)):
            return False
        if not np.isclose(self.phase_constant.real, self.phase_constant):
            return False
        return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            np.array_equal(self.parity_matrix, other.parity_matrix)
            and np.array_equal(self.phase_matrix, other.phase_matrix)
            and np.array_equal(self.parity_shift, other.parity_shift)
            and np.array_equal(self.phase_shift, other.phase_shift)
            and np.isclose(abs(self.phase_constant - other.phase_constant) % 4, 0)
        )

    @property
    def changes_parity(self):
        return self.parity_shift.any() or not np.array_equal(
            self.parity_matrix, np.eye(self._num_qubits)
        )

    @property
    def changes_phase(self):
        return self.phase_shift.any() or self.phase_matrix.any()

    def __matmul__(self, other) -> "HadamardFreeGate":
        if not isinstance(other, type(self)):
            return NotImplemented

        if self._num_qubits != other._num_qubits:
            return NotImplemented

        G2, G1 = self, other
        assert G2.validate()
        assert G1.validate()
        c = G2.phase_constant + G1.phase_constant
        if not G2.changes_parity:
            Δ = np.copy(G1.parity_matrix)
            Γ = G1.phase_matrix + G2.phase_matrix
            X = np.copy(G1.parity_shift)
            Z = G1.phase_shift + G2.phase_shift + (np.diag(Γ) // 2)
        elif not G1.changes_phase:
            Δ = G2.parity_matrix @ G1.parity_matrix
            Γ = np.copy(G2.phase_matrix)
            X = G2.parity_matrix @ G1.parity_shift + G2.parity_shift
            Z = np.copy(G2.phase_shift)
        elif not (G2.changes_phase or G1.changes_parity):
            Δ = np.copy(G2.parity_matrix)
            Δ2_T_inv = linalg.get_inverse(Δ.T)
            Δ2_inv = linalg.get_inverse(Δ)

            Γ = Δ2_T_inv @ G1.phase_matrix @ Δ2_inv
            Z = Δ2_T_inv @ G1.phase_shift + Γ.T @ G2.parity_shift + (np.diag(Γ) // 2)
            Γ %= 2
            X = np.copy(G2.parity_shift)
            c -= (G2.parity_shift @ np.dot(Γ, G2.parity_shift)) % 4
            c -= (2 * Z @ G2.parity_shift) % 4
        else:
            C2, B2 = G2.to_phase_then_parity()
            assert C2.validate()
            assert B2.validate()
            assert B2 @ C2 == G2
            R = C2 @ G1
            assert R.validate()
            C1, B1 = R.to_phase_then_parity()
            assert C1.validate()
            assert B1.validate()
            assert (B1 @ C1) == R
            return (B2 @ B1) @ C1

        return type(self)(
            num_qubits=self._num_qubits,
            parity_matrix=Δ % 2,
            phase_matrix=Γ % 2,
            parity_shift=X % 2,
            phase_shift=Z % 2,
            phase_constant=c % 4,
        )

    @classmethod
    def random(cls, num_qubits: int, seed=None, include_phase_constant=True):
        rng = np.random.default_rng(seed)
        return cls(
            num_qubits=num_qubits,
            parity_matrix=random.random_invertible_matrix(num_qubits, seed=rng),
            phase_matrix=random.random_symmetric_matrix(num_qubits, seed=rng),
            parity_shift=random.random_vector(num_qubits, seed),
            phase_shift=random.random_vector(num_qubits, seed),
            phase_constant=4 * rng.random(),
        )

    def _args_str(self, include_num_qubits: bool = True, suffix: str = ""):
        args = [f"n={self._num_qubits}"] if include_num_qubits else []
        if not np.array_equal(self.parity_matrix, np.eye(self._num_qubits)):
            args.append(f"Δ{suffix}={linalg.matrix_to_tight_str(self.parity_matrix)}")
        if self.phase_matrix.any():
            args.append(f"Γ{suffix}={linalg.matrix_to_tight_str(self.phase_matrix)}")
        if self.parity_shift.any():
            args.append(f"X{suffix}={linalg.vector_to_tight_str(self.parity_shift)}")
        if self.phase_shift.any():
            args.append(f"Z{suffix}={linalg.vector_to_tight_str(self.phase_shift)}")
        return ", ".join(args)

    def __str__(self):
        return (
            "" if np.isclose(self.phase_constant, 0) else f"i^{self.phase_constant} * "
        ) + f"{type(self).__name__}({self._args_str()})"

    def inverse(self):
        if not self.changes_parity:
            Γ = np.copy(self.phase_matrix)
            Z = self.phase_shift + np.diag(Γ)
            return type(self)(
                num_qubits=self._num_qubits,
                phase_matrix=Γ,
                phase_shift=Z % 2,
                phase_constant=-self.phase_constant,
            )
        if not self.changes_phase:
            Δ = linalg.get_inverse(self.parity_matrix)
            X = (Δ @ self.parity_shift) % 2
            return type(self)(
                num_qubits=self._num_qubits,
                parity_matrix=Δ,
                parity_shift=X,
                phase_constant=-self.phase_constant,
            )
        B, C = self.split()
        return B.inverse() @ C.inverse()

    def is_minimal(self, hadamards: np.ndarray):
        """TODO"""
        if self.parity_shift.any():
            return False

        H_indices = np.flatnonzero(hadamards)
        k = len(H_indices)
        I_indices = np.flatnonzero(np.logical_not(hadamards))
        HH_indices = np.ix_(H_indices, H_indices)
        HI_indices = np.ix_(H_indices, I_indices)
        IH_indices = np.ix_(I_indices, H_indices)
        II_indices = np.ix_(I_indices, I_indices)
        if not np.array_equal(self.parity_matrix[HH_indices], np.eye(k)):
            return False
        if not np.array_equal(
            self.parity_matrix[II_indices], np.eye(self._num_qubits - k)
        ):
            return False
        if self.parity_matrix[np.ix_(H_indices, I_indices)].any():
            return False

        if self.phase_shift[I_indices].any():
            return False
        if self.phase_matrix[HI_indices].any():
            return False
        if self.phase_matrix[IH_indices].any():
            return False
        if self.phase_matrix[II_indices].any():
            return False

        return True

    def split(self) -> Tuple[Any, Any]:
        return type(self)(
            num_qubits=self._num_qubits,
            parity_matrix=np.copy(self.parity_matrix),
            parity_shift=np.copy(self.parity_shift),
        ), type(self)(
            num_qubits=self._num_qubits,
            phase_matrix=np.copy(self.phase_matrix),
            phase_shift=np.copy(self.phase_shift),
            phase_constant=self.phase_constant,
        )

    def to_phase_then_parity(self) -> Tuple[Any, Any]:
        new_Γ = self.parity_matrix.T @ self.phase_matrix @ self.parity_matrix
        new_Z = (np.diag(new_Γ) // 2) + (
            self.parity_matrix.T
            @ (self.phase_shift + (self.phase_matrix @ self.parity_shift))
        )
        new_phase_constant = (
            self.phase_constant
            + 2 * self.parity_shift @ self.phase_shift
            + self.parity_shift @ self.phase_matrix @ self.parity_shift
        ) % 4
        return type(self)(
            num_qubits=self._num_qubits, phase_matrix=new_Γ % 2, phase_shift=new_Z % 2
        ), type(self)(
            num_qubits=self._num_qubits,
            parity_matrix=np.copy(self.parity_matrix),
            parity_shift=np.copy(self.parity_shift),
            phase_constant=new_phase_constant,
        )

    def unitary(self):
        N = 2**self._num_qubits
        input_bitstrings = indexing.get_all_bitstrings(self._num_qubits)
        output_bitstrings = (
            self.parity_matrix @ input_bitstrings.transpose((1, 0))
        ).transpose((1, 0)) % 2
        output_bitstrings ^= self.parity_shift[np.newaxis, :]
        output_indices = indexing.bitstrings_to_indices(output_bitstrings)
        U = np.zeros((N, N), dtype=np.complex128)
        phases = np.einsum(
            "ij,jk,ik->i", output_bitstrings, self.phase_matrix, output_bitstrings
        ) + (2 * output_bitstrings @ self.phase_shift)
        U[output_indices, np.arange(N)] = 1j**phases
        return (1j**self.phase_constant) * U
