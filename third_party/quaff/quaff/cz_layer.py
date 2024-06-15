import collections
import itertools
from typing import Iterable, Tuple, cast

import cirq
import numpy as np
from quaff import basis_change, gates, indexing, linalg


def get_index_range(num_qubits: int, num_steps: int, i: int):
    num_steps %= 2 * (num_qubits + 1)
    if num_steps == 0:
        return i, i
    if num_steps == num_qubits + 1:
        return (num_qubits - 1 - i,) * 2
    if num_steps > num_qubits + 1:
        raise NotImplementedError

    min_index = i - num_steps + ((i + num_steps) % 2)
    if i < num_steps - 1:
        min_index = -1 - min_index
    min_index %= num_qubits

    max_index = num_steps + i - 1 + ((i + 1 + num_steps) % 2)
    if max_index >= num_qubits:
        max_index = -1 - max_index
    max_index %= num_qubits
    return min_index, max_index


class CZLayerBasisChangeStageGate(cirq.Gate):
    def __init__(self, num_qubits: int, num_steps: int = 2):
        self.num_steps = int(num_steps)
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    @property
    def basis_change_matrix(self):
        return self.get_basis_change_matrix(self._num_qubits, self.num_steps)

    @staticmethod
    def get_basis_change_matrix(num_qubits: int, num_steps: int = 2):
        matrix = np.zeros((num_qubits,) * 2, dtype=linalg.DTYPE)
        for i in range(num_qubits):
            min_index, max_index = get_index_range(num_qubits, num_steps, i)
            matrix[i, np.arange(min_index, max_index + 1)] = 1
        return matrix

    def as_basis_change_gate(self):
        return basis_change.BasisChangeGate(self.basis_change_matrix)

    def _decompose_(self, qubits):
        n = self.num_qubits()
        for s in range(self.num_steps):
            if not (s % 2):
                for i in range(0, n - 1, 2):
                    yield cirq.CNOT(qubits[i], qubits[i + 1])
                for i in range(1, n - 1, 2):
                    yield cirq.CNOT(qubits[i + 1], qubits[i])
            else:
                for i in range(0, n - 1, 2):
                    yield cirq.CNOT(qubits[i + 1], qubits[i])
                for i in range(1, n - 1, 2):
                    yield cirq.CNOT(qubits[i], qubits[i + 1])

    def _unitary_(self):
        return self.as_basis_change_gate()._unitary_()

    def __str__(self):
        return f"{type(self).__name__}({self._num_qubits}, {self.num_steps})"


class CZLayerGate(cirq.Gate):
    def __init__(
        self,
        num_qubits: int,
        pairs: Iterable[Tuple[int, int]],
        include_first_stage: bool = True,
        include_final_stage: bool = True,
    ):
        counter = collections.Counter(tuple(sorted(pair)) for pair in pairs)
        pairs = tuple(
            cast(Tuple[int, int], pair) for pair in sorted(counter) if counter[pair] % 2
        )
        for pair in pairs:
            if len(set(pair)) != 2:
                raise ValueError(
                    f"Pair {pair} is does not contain exactly two distinct elements."
                )
            for i in pair:
                if not (0 <= i < num_qubits):
                    raise ValueError(f"Index {i} out of range [0, {num_qubits}).")

        self._num_qubits = num_qubits
        self.pairs = pairs
        if not (include_first_stage or include_final_stage):
            raise ValueError(
                "At most one of the first and final stages can be omitted."
            )
        if not (include_first_stage and include_final_stage):
            raise NotImplementedError
        self.includes_first_stage = bool(include_first_stage)
        self.includes_final_stage = bool(include_final_stage)

    def __str__(self):
        pair_str = ";".join(",".join(str(i) for i in pair) for pair in self.pairs)
        return f"CZs({pair_str})"

    def num_qubits(self):
        return self._num_qubits

    def _unitary_(self):
        n = self.num_qubits()
        basis = indexing.get_all_bitstrings(n)
        parity = np.zeros(2**n, dtype=linalg.DTYPE)
        for pair in self.pairs:
            parity += (basis[:, pair] == 1).all(axis=1)
        U = np.diag((-1) ** parity)
        axes = np.concatenate((np.arange(n)[::-1], np.arange(n, 2 * n)))
        return np.transpose(U.reshape((2,) * 2 * n), axes).reshape((2**n,) * 2)

    @property
    def phase_poly(self):
        return self.get_phase_poly(self.num_qubits(), self.pairs)

    @staticmethod
    def get_phase_poly(num_qubits, pairs):
        phase_poly = np.zeros((num_qubits,) * 2, dtype=linalg.DTYPE)
        for pair in pairs:
            i, j = sorted(pair)
            assert i < j
            phase_poly[i, i] += 1
            phase_poly[j, j] += 1
            if j == i + 1:
                phase_poly[i, j] -= 1
                continue
            assert i + 1 <= j - 1
            phase_poly[i, i] += 1
            phase_poly[j, j] += 1
            for a, b in itertools.product((i, i + 1), (j - 1, j)):
                phase_poly[a, b] += 1
        return phase_poly % 4

    def _get_phases(self):
        """TODO"""
        n = self.num_qubits()
        phase_poly = self.phase_poly
        quadratic = np.zeros((n // 2, n), dtype=linalg.DTYPE)
        for s in range(n // 2):
            for r in range(n):
                i, j = get_index_range(n, 2 * (s + 1), r)
                if i == j:
                    continue
                quadratic[s, r] = (phase_poly[i, j] + phase_poly[j, i]) % 4
        return np.diag(phase_poly), quadratic

    def _decompose_(self, qubits):
        n = self.num_qubits()

        basis_change_gate = CZLayerBasisChangeStageGate(n, 2)

        linear, quadratic = self._get_phases()
        single_qubit_phase_layer = gates.SingleQubitLayerGate(
            cirq.S**u if u else cirq.I for u in linear
        )(*qubits)

        yield single_qubit_phase_layer

        for s in range(n // 2):
            yield basis_change_gate(*qubits)
            for q, u in zip(qubits, quadratic[s]):
                if u:
                    yield (cirq.S**u)(q)

        if self.includes_final_stage:
            yield CZLayerBasisChangeStageGate(n, 1 + (n % 2))(*qubits)
