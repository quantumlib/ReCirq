import dataclasses
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import cirq
import numpy as np
from quaff import linalg, sampling, testing
from quaff.gates.hadamard_free import HadamardFreeGate
from quaff.gates.single_qubit_layer import SingleQubitLayerGate


def _recover_permutation(matrix: np.ndarray) -> Optional[np.ndarray]:
    """If matrix is permutation of a lower-triangular, unit-diagonal matrix,
    return the permutation. If not, then output is undetermined"""
    # TODO: make sure output is always determined
    n = len(matrix)
    S_inv = n - 1 - np.argmax(np.flip(matrix, 1), 1)
    if not np.array_equal(sorted(S_inv), range(n)):
        return None
    return linalg.invert_permutation(S_inv)


@dataclass
class FHFGate(cirq.Gate):
    num_qubits: dataclasses.InitVar[int]
    _num_qubits: int = dataclasses.field(init=False)

    F1: HadamardFreeGate = dataclasses.field(init=False)
    H: np.ndarray = dataclasses.field(init=False)
    F2: HadamardFreeGate = dataclasses.field(init=False)

    first_H_free_gate: dataclasses.InitVar[HadamardFreeGate] = None
    hadamards: dataclasses.InitVar[np.ndarray] = None
    second_H_free_gate: dataclasses.InitVar[np.ndarray] = None

    def __post_init__(
        self, num_qubits, first_H_free_gate, hadamards, second_H_free_gate
    ):
        self._num_qubits = num_qubits
        if first_H_free_gate is None:
            self.F1 = HadamardFreeGate(num_qubits)
        else:
            self.F1 = first_H_free_gate
        if hadamards is None:
            self.H = np.zeros(num_qubits, dtype=linalg.DTYPE)
        else:
            self.H = np.array(hadamards, dtype=linalg.DTYPE)
        if second_H_free_gate is None:
            self.F2 = HadamardFreeGate(num_qubits)
        else:
            self.F2 = second_H_free_gate

    @classmethod
    def _from_json_dict_(cls, _num_qubits, F1, H, F2, **kwargs):
        return cls(
            num_qubits=_num_qubits,
            first_H_free_gate=F1,
            hadamards=H,
            second_H_free_gate=F2,
        )

    def _json_dict_(self):
        print(cirq.dataclass_json_dict(self))
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return "quaff"

    def _num_qubits_(self):
        return self._num_qubits

    @property
    def phase_constant(self):
        return (self.F1.phase_constant + self.F2.phase_constant) % 4

    def __str__(self):
        args_str = ", ".join(
            (
                self.F1._args_str(suffix="1"),
                #               f"Δ1={linalg.matrix_to_tight_str(self.F1.parity_matrix)}",
                #               f"Γ1={linalg.matrix_to_tight_str(self.F1.phase_matrix)}",
                #               f"X1={linalg.vector_to_tight_str(self.F1.parity_shift)}",
                #               f"Z1={linalg.vector_to_tight_str(self.F1.phase_shift)}",
                f"H={linalg.vector_to_tight_str(self.H)}",
                self.F2._args_str(include_num_qubits=False, suffix="1"),
                #               f"Δ2={linalg.matrix_to_tight_str(self.F2.parity_matrix)}",
                #               f"Γ2={linalg.matrix_to_tight_str(self.F2.phase_matrix)}",
                #               f"X2={linalg.vector_to_tight_str(self.F2.parity_shift)}",
                #               f"Z2={linalg.vector_to_tight_str(self.F2.phase_shift)}",
            )
        )

        return (
            "" if np.isclose(self.phase_constant, 0) else f"i^{self.phase_constant} * "
        ) + f"FHFGate({args_str})"

    def validate(self) -> bool:
        return (
            self.F1.validate()
            and testing.is_boolean_array_of_shape(self.H, (self._num_qubits,))
            and self.F2.validate()
        )

    def is_canonical(self) -> bool:
        if not np.array_equal(self.F1.phase_matrix, self.F1.phase_matrix.T):
            return False

        S = cast(np.ndarray, _recover_permutation(self.F1.parity_matrix))
        if not linalg.is_tril_and_unit_diag(self.F1.parity_matrix[S]):
            return False

        if self.F2.parity_shift.any():
            return False
        if self.F2.phase_shift.any():
            return False

        if not np.array_equal(self.F2.phase_matrix, self.F2.phase_matrix.T):
            return False

        for i, j in itertools.product(range(self._num_qubits), repeat=2):
            if (
                ((not self.H[i]) and (not self.H[j]) and self.F2.phase_matrix[i, j])
                or (
                    self.H[i]
                    and (not self.H[j])
                    and S[i] > S[j]
                    and self.F2.phase_matrix[i, j]
                )
                or (
                    (not self.H[i])
                    and (not self.H[j])
                    and S[i] > S[j]
                    and self.F2.parity_matrix[i, j]
                )
                or (
                    self.H[i]
                    and self.H[j]
                    and S[i] < S[j]
                    and self.F2.parity_matrix[i, j]
                )
                or (self.H[i] and (not self.H[j]) and self.F2.parity_matrix[i, j])
            ):
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._num_qubits == other._num_qubits
            and self.F1 == other.F1
            and np.array_equal(self.H, other.H)
            and self.F2 == other.F2
        )

    def copy(self):
        return type(self)(
            num_qubits=self._num_qubits,
            first_H_free_gate=self.F1.copy(),
            hadamards=np.copy(self.H),
            second_H_free_gate=self.F2.copy(),
        )

    def _hadamard_layer_gate(self):
        return SingleQubitLayerGate(
            cast(cirq.testing.SingleQubitGate, cirq.H if h else cirq.I)
            for i, h in enumerate(self.H)
        )

    def split(self) -> Tuple[cirq.Gate, ...]:
        return self.F1, self._hadamard_layer_gate(), self.F2

    def push_earlier(self) -> None:
        n = self._num_qubits
        F0 = self.F1
        assert F0.validate()
        self.F1 = type(F0)(num_qubits=n)

        if self.F2.parity_shift.any():
            raise NotImplementedError
        if self.F2.phase_shift.any():
            raise NotImplementedError

        H_indices = np.flatnonzero(self.H)
        I_indices = np.flatnonzero(np.logical_not(self.H))
        HH_indices = np.ix_(H_indices, H_indices)
        HI_indices = np.ix_(H_indices, I_indices)
        IH_indices = np.ix_(I_indices, H_indices)
        II_indices = np.ix_(I_indices, I_indices)

        k = len(H_indices)
        assert k == self.H.sum()
        assert k + len(I_indices) == n
        Δ_HH = self.F2.parity_matrix[HH_indices]
        Δ_HI = self.F2.parity_matrix[HI_indices]
        Δ_IH = self.F2.parity_matrix[IH_indices]
        Δ_II = self.F2.parity_matrix[II_indices]
        if np.any(Δ_HI):
            raise NotImplementedError

        I_HH = np.eye(k, dtype=linalg.DTYPE)
        I_HI = np.zeros_like(Δ_HI)
        I_IH = np.zeros_like(Δ_IH)
        I_II = np.eye(n - k, dtype=linalg.DTYPE)

        # zero out blocks of parity matrix other than IH
        Δ_HH_inv = cast(np.ndarray, linalg.get_inverse(Δ_HH, True))
        new_Δ_IH = (Δ_IH @ Δ_HH_inv) % 2

        Δ = np.block([[Δ_HH, Δ_HI], [Δ_IH, Δ_II]])
        new_Δ = np.block([[I_HH, 0 * I_HI], [new_Δ_IH, I_II]])
        eye = np.eye(n, dtype=linalg.DTYPE)
        assert np.array_equal(self.F1.parity_matrix, eye)
        new_Δ_HH = linalg.get_inverse(Δ_HH.T)
        assert np.array_equal(new_Δ_HH, Δ_HH_inv.T)
        self.F1.parity_matrix[HH_indices] = new_Δ_HH
        self.F1.parity_matrix[II_indices] = Δ_II
        self.F2.parity_matrix = np.copy(eye)
        self.F2.parity_matrix[IH_indices] = new_Δ_IH

        # zero out II block of phase matrix
        assert self.F1.validate()
        assert F0.validate()
        F0 = self.F1 @ F0
        self.F1 = type(F0)(num_qubits=n)
        C, B = self.F2.to_phase_then_parity()
        assert not self.F1.phase_matrix[II_indices].any()
        assert not self.F1.phase_shift[I_indices].any()
        self.F1.phase_matrix[II_indices] = C.phase_matrix[II_indices]
        self.F1.phase_shift[I_indices] = C.phase_shift[I_indices]
        C.phase_matrix[II_indices] = 0
        C.phase_shift[I_indices] = 0

        # zero out IH, HI blocks of phase matrix
        assert np.array_equal(self.F1.parity_matrix, np.eye(n))
        assert not self.F1.parity_shift.any()
        assert np.array_equal(C.phase_matrix[HI_indices], C.phase_matrix[IH_indices].T)
        self.F1.parity_matrix[HI_indices] = C.phase_matrix[HI_indices]
        C.phase_matrix[HI_indices] = 0
        C.phase_matrix[IH_indices] = 0
        self.F2 = B @ C

        self.F1 @= F0

    def _decompose_(self, qubits):
        for gate in self.split():
            yield gate(*qubits)

    @classmethod
    def _from_sample(cls, sample: sampling.CliffordSample):
        num_qubits = sample.num_qubits()
        inv_perm = linalg.invert_permutation(sample.permutation)

        first_H_free_gate = HadamardFreeGate(
            num_qubits=num_qubits,
            parity_matrix=np.array(sample.first_parity_matrix, dtype=linalg.DTYPE)[
                inv_perm
            ],
            phase_matrix=np.array(sample.first_phase_matrix, dtype=linalg.DTYPE)[
                np.ix_(inv_perm, inv_perm)
            ],
            parity_shift=np.array(sample.parity_shift, dtype=linalg.DTYPE)[inv_perm],
            phase_shift=np.array(sample.phase_shift, dtype=linalg.DTYPE)[inv_perm],
        )

        second_H_free_gate = HadamardFreeGate(
            num_qubits=num_qubits,
            parity_matrix=np.array(sample.second_parity_matrix, dtype=linalg.DTYPE),
            phase_matrix=np.array(sample.second_phase_matrix, dtype=linalg.DTYPE),
        )

        FHF = cls(
            num_qubits,
            first_H_free_gate=first_H_free_gate,
            hadamards=np.array(sample.hadamards, dtype=linalg.DTYPE),
            second_H_free_gate=second_H_free_gate,
        )
        return FHF

    @classmethod
    def random(cls, num_qubits, seed=None):
        rng = np.random.default_rng(seed)
        sample = sampling.CliffordSampler(num_qubits).sample(rng)
        return cls._from_sample(sample)
