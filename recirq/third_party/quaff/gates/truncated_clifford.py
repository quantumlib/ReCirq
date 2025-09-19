import dataclasses
from dataclasses import dataclass
from typing import cast

import cirq
import numpy as np
from recirq.third_party.quaff import comb, cz_layer, indexing, linalg, sampling, testing
from recirq.third_party.quaff.gates.fhf import FHFGate
from recirq.third_party.quaff.gates.hadamard_free import HadamardFreeGate
from recirq.third_party.quaff.gates.single_qubit_layer import SingleQubitLayerGate
from recirq.third_party.quaff.linalg import BooleanMatrix, BooleanVector


@dataclass(eq=True, order=True)
class TruncatedCliffordGate(cirq.Gate):
    num_qubits: dataclasses.InitVar[int]
    _num_qubits: int = dataclasses.field(init=False)
    H: BooleanVector = cast(BooleanVector, None)
    CX: BooleanMatrix = cast(BooleanMatrix, None)
    CZ: BooleanVector = cast(BooleanVector, None)
    P: BooleanVector = cast(BooleanVector, None)

    def __post_init__(self, num_qubits):
        self._num_qubits = num_qubits
        if self.H is None:
            self.H = (0,) * num_qubits
        else:
            self.H = tuple(linalg.DTYPE(x) for x in self.H)
        k = self.rank
        if self.CX is None:
            if k:
                self.CX = ((0,) * k,) * (num_qubits - k)
            else:
                self.CX = ()
        else:
            self.CX = linalg.tuple_of_tuples(self.CX)
        if self.CZ is None:
            self.CZ = (0,) * comb.binom(k)
        else:
            self.CZ = tuple(linalg.DTYPE(x) for x in self.CZ)
        if self.P is None:
            self.P = (0,) * k
        else:
            self.P = tuple(linalg.DTYPE(x) for x in self.P)

    @classmethod
    def _from_json_dict_(cls, *, _num_qubits, **kwargs):
        del kwargs["cirq_type"]
        return cls(num_qubits=_num_qubits, **kwargs)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return "quaff"

    def __str__(self):
        args = [f"n={self._num_qubits}"]
        if np.any(self.H):
            args.append(f"H={linalg.vector_to_tight_str(self.H)}")
        if np.any(self.CX):
            args.append(f"CX={linalg.matrix_to_tight_str(self.CX)}")
        if np.any(self.CZ) or np.any(self.P):
            args.append(
                f"phase={linalg.matrix_to_tight_str(self._phase_matrix_block())}"
            )
        args_str = ", ".join(args)
        return f"{type(self).__name__}({args_str})"

    def copy(self):
        return type(self)(
            num_qubits=self._num_qubits, H=self.H, CX=self.CX, CZ=self.CZ, P=self.P
        )

    @property
    def rank(self):
        return sum(self.H)

    def validate(self) -> bool:
        k = self.rank
        if not testing.is_boolean_tuple_of_len(self.P, k):
            return False
        if not testing.is_boolean_tuple_of_len(self.CZ, comb.binom(k)):
            return False
        if not testing.is_boolean_tuple_of_len(self.H, self._num_qubits):
            return False
        if k in (0, self._num_qubits):
            if self.CX != ():
                return False
        else:
            if not testing.is_nested_boolean_tuples_of_shape(
                self.CX, (self._num_qubits - k, k)
            ):
                return False
        return True

    def _num_qubits_(self):
        return self._num_qubits

    def _phase_matrix_block(self) -> np.ndarray:
        k = self.rank
        Γ = np.zeros((k, k), dtype=linalg.DTYPE)
        Γ[indexing.triu_indices(k)] = self.CZ
        Γ[indexing.tril_indices(k)] = self.CZ
        Γ[np.diag_indices(k)] = self.P
        return Γ

    #   def unitary(self) -> np.ndarray:
    #       H_indices = np.flatnonzero(self.H)
    #       I_indices = np.flatnonzero(np.logical_not(self.H))
    #       n = self.num_qubits
    #       N = 2 ** n

    #       H_bits = indexing.get_all_bitstrings(self.num_qubits)[:, H_indices]
    #       Γ = self._phase_matrix_block()
    #       phases = np.einsum("ij,jk,ik->i", H_bits, Γ, H_bits) % 4
    #       bitstrings = indexing.get_all_bitstrings(self.num_qubits)

    #       Δ = np.eye(n, dtype=linalg.DTYPE)
    #       if self.rank not in (0, n):
    #           Δ[np.ix_(I_indices, H_indices)] = self.CX
    #       output_bitstrings = (bitstrings @ Δ.T) % 2
    #       output_indices = indexing.bitstrings_to_indices(output_bitstrings)

    #       U = np.zeros((N, N), dtype=np.complex128)
    #       U[output_indices, np.arange(N)] = 1j ** phases
    #       return U

    def _CZ_layer(self):
        H_indices = np.flatnonzero(self.H)
        I_indices = np.flatnonzero(np.logical_not(self.H))
        pairs = []
        for i, j, cz in zip(*indexing.triu_indices(self.rank), self.CZ):
            if cz:
                pairs.append((H_indices[i], H_indices[j]))
        for i, row in zip(I_indices, self.CX):
            for j, cx in zip(H_indices, row):
                if cx:
                    pairs.append((i, j))

        return cz_layer.CZLayerGate(num_qubits=self._num_qubits, pairs=pairs)

    def _full_P(self):
        P = np.zeros(self._num_qubits, dtype=linalg.DTYPE)
        H_indices = np.flatnonzero(self.H)
        P[H_indices] = self.P
        return P

    def _decompose_(self, qubits):
        first_layer = [cirq.I] * self._num_qubits
        j = 0
        for i, h in enumerate(self.H):
            if h:
                if self.P[j]:
                    first_layer[i] = cirq.S
                j += 1
            else:
                first_layer[i] = cirq.H

        yield SingleQubitLayerGate(first_layer)(*qubits)
        yield self._CZ_layer()(*qubits)
        yield SingleQubitLayerGate(cirq.H for _ in self.H)(*qubits)

    def to_FHF_gate(self):
        n = self._num_qubits
        H_indices = np.flatnonzero(self.H)
        I_indices = np.flatnonzero(np.logical_not(self.H))

        Δ = np.eye(n, dtype=linalg.DTYPE)
        if self.rank not in (0, n):
            Δ[np.ix_(I_indices, H_indices)] = self.CX

        Γ = np.zeros((n, n), dtype=linalg.DTYPE)
        if self.rank:
            Γ[np.ix_(H_indices, H_indices)] = self._phase_matrix_block()

        F1 = HadamardFreeGate(num_qubits=n, parity_matrix=Δ, phase_matrix=Γ)

        F2 = HadamardFreeGate(
            num_qubits=n, parity_matrix=np.flip(np.eye(n, dtype=linalg.DTYPE), 0)
        )

        return FHFGate(
            num_qubits=n, first_H_free_gate=F1, hadamards=self.H, second_H_free_gate=F2
        )

    @classmethod
    def _from_sample(cls, sample: sampling.CliffordSample):
        FHF_gate = FHFGate._from_sample(sample)
        FHF_gate.push_earlier()

        F2, H = FHF_gate.F2, FHF_gate.H

        H_indices = np.flatnonzero(H)
        I_indices = np.flatnonzero(np.logical_not(H))
        HH_indices = np.ix_(H_indices, H_indices)
        IH_indices = np.ix_(I_indices, H_indices)

        k = len(H_indices)

        CX = F2.parity_matrix[IH_indices]
        CZ = F2.phase_matrix[HH_indices][indexing.tril_indices(k)]
        P = np.diag(F2.phase_matrix)[H_indices]

        return cls(
            num_qubits=sample.num_qubits(),
            H=tuple(H),
            CX=tuple(tuple(row) for row in CX) if CX.size else (),
            CZ=tuple(CZ),
            P=tuple(P),
        )

    @classmethod
    def random(cls, num_qubits, seed=None):
        rng = np.random.default_rng(seed)
        sample = sampling.CliffordSampler(num_qubits).sample(rng)
        return cls._from_sample(sample)
