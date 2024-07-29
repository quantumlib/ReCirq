import itertools
import math
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

import cirq
import numpy as np
from quaff import comb, indexing, linalg, testing
from quaff.sampling.sampler import SingleParameterSampler

QuantumMallowsSample = Tuple[Tuple[bool, ...], Tuple[int, ...]]
QuantumMallowsRandomness = Tuple[int, ...]


class QuantumMallowsSampler(SingleParameterSampler):
    def sample_randomness(
        self, rng: Optional[np.random.Generator] = None
    ) -> QuantumMallowsRandomness:
        rng = np.random.default_rng(rng)
        return tuple(rng.integers(0, 4**m - 1) for m in np.arange(self.n, 0, -1))

    def randomness_iter(self) -> Iterable[QuantumMallowsRandomness]:
        return itertools.product(*(range(k) for k in 4 ** np.arange(self.n, 0, -1) - 1))

    def randomness_size(self) -> int:
        return math.prod(4 ** np.arange(self.n, 0, -1) - 1)

    def validate_randomness(self, randomness: QuantumMallowsRandomness) -> bool:
        if any(r < 0 for r in randomness):
            return False
        return all(r < 4**m - 1 for r, m in zip(randomness, np.arange(self.n, 0, -1)))

    def _parse_randomness(self, randomness) -> Iterable[Tuple[int, int, int, int]]:
        for i, r in enumerate(randomness):
            m = self.n - i
            threshold = 2**m - 1
            if r < threshold:
                h = 0
                threshold = 0
                for k in range(1, m + 1):
                    threshold += 2 ** (k - 1)
                    if r < threshold:
                        break
            else:
                h = 1
                r -= threshold
                threshold = 0
                for k in range(1, m + 1):
                    threshold += 2 ** (2 * m - k)
                    if r < threshold:
                        break
            yield i, k - 1, h, threshold - 1 - r

    def randomness_to_sample(
        self, randomness: QuantumMallowsRandomness
    ) -> QuantumMallowsSample:
        H = np.zeros(self.n, dtype=bool)
        S = np.zeros(self.n, dtype=int)
        A = list(range(self.n))
        for i, k, h, _ in self._parse_randomness(randomness):
            H[i] = h
            S[i] = A.pop(k)

        return tuple(H), tuple(S)

    def num_unique_samples(self):
        return (2**self.n) * math.factorial(self.n)

    def unique_samples_iter(self) -> Iterable[QuantumMallowsSample]:
        return itertools.product(
            itertools.product((False, True), repeat=self.n),
            itertools.permutations(range(self.n)),
        )

    def sample_multiplicity(self, sample: QuantumMallowsSample):
        h, S = sample
        exponent = (
            comb.binom(self.n)
            + np.sum(h)
            + sum(
                1 if h[i] else -1
                for i, j in itertools.combinations(range(self.n), 2)
                if S[i] < S[j]
            )
        )
        return 2**exponent

    def name(self):
        return "QuantumMallowsSampler"


@dataclass(frozen=True, order=True)
class CliffordSample:
    first_parity_matrix: linalg.BooleanMatrix
    first_phase_matrix: linalg.BooleanMatrix
    parity_shift: linalg.BooleanVector
    phase_shift: linalg.BooleanVector
    permutation: Tuple[int, ...]
    hadamards: linalg.BooleanVector
    second_parity_matrix: linalg.BooleanMatrix
    second_phase_matrix: linalg.BooleanMatrix

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls(
            **{
                key: (
                    linalg.tuple_of_tuples(val)
                    if key.endswith("matrix")
                    else tuple(val)
                )
                for key, val in kwargs.items()
                if key != "cirq_type"
            }
        )

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return "quaff"

    def num_qubits(self):
        return len(self.parity_shift)

    def __str__(self):
        args_str = ", ".join(
            (
                f"Δ1={linalg.matrix_to_tight_str(self.first_parity_matrix)}",
                f"Γ1={linalg.matrix_to_tight_str(self.first_phase_matrix)}",
                f"X={linalg.vector_to_tight_str(self.parity_shift)}",
                f"Z={linalg.vector_to_tight_str(self.phase_shift)}",
                f"S={self.permutation}",
                f"H={linalg.vector_to_tight_str(self.hadamards)}",
                f"Δ2={linalg.matrix_to_tight_str(self.second_parity_matrix)}",
                f"Γ2={linalg.matrix_to_tight_str(self.second_phase_matrix)}",
            )
        )
        return f"CliffordSample({args_str})"


class CliffordRandomness(NamedTuple):
    mallows_randomness: QuantumMallowsRandomness
    parity_matrix_randomness: Tuple[bool, ...]
    phase_matrix_randomness: Tuple[bool, ...]
    parity_shift_randomness: Tuple[bool, ...]
    phase_shift_randomness: Tuple[bool, ...]


class CliffordSampler(SingleParameterSampler):
    def __init__(self, n: int):
        self.n = n
        self.mallows_sampler = QuantumMallowsSampler(n)

    def sample_randomness(
        self, rng: Optional[np.random.Generator]
    ) -> CliffordRandomness:
        rng = np.random.default_rng(rng)
        binom = comb.binom(self.n)
        return CliffordRandomness(
            mallows_randomness=self.mallows_sampler.sample_randomness(rng),
            parity_matrix_randomness=tuple(rng.integers(0, 2, binom)),
            phase_matrix_randomness=tuple(rng.integers(0, 2, binom + self.n)),
            parity_shift_randomness=tuple(rng.integers(0, 2, self.n)),
            phase_shift_randomness=tuple(rng.integers(0, 2, self.n)),
        )

    def _get_Δ_and_Γ_indices(
        self, H: np.ndarray, S: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Δ_indices, Γ_indices = [], []
        for j, i in itertools.combinations(range(self.n), 2):
            if H[j]:
                if H[i]:
                    Γ_indices.append((i, j))
                    if S[i] > S[j]:
                        Δ_indices.append((i, j))
                else:
                    Δ_indices.append((i, j))
                    if S[i] > S[j]:
                        Γ_indices.append((i, j))
            else:
                if S[i] >= S[j]:
                    continue
                if H[i]:
                    Γ_indices.append((i, j))
                else:
                    Δ_indices.append((i, j))
        return (
            (
                tuple(np.array(Δ_indices).T)
                if Δ_indices
                else (np.zeros(0, dtype=int),) * 2
            ),
            (
                tuple(np.array(Γ_indices).T)
                if Γ_indices
                else (np.zeros(0, dtype=int),) * 2
            ),
        )

    def randomness_to_sample(self, randomness: CliffordRandomness) -> CliffordSample:
        S = np.zeros(self.n, dtype=int)
        A = list(range(self.n))
        H = np.zeros(self.n, dtype=linalg.DTYPE)

        Δ1, Δ2 = (np.eye(self.n, dtype=linalg.DTYPE) for _ in "12")
        Γ1, Γ2 = (np.zeros((self.n,) * 2, dtype=linalg.DTYPE) for _ in "12")

        Γ1_diag_bits = randomness.phase_matrix_randomness[: self.n]
        Γ1[np.diag_indices(self.n)] = Γ1_diag_bits

        Γ1_offdiag_bits = randomness.phase_matrix_randomness[self.n :]
        if self.n > 1:
            tril_indices = indexing.tril_indices(self.n)
            triu_indices = indexing.triu_indices(self.n)
            Δ1[tril_indices] = randomness.parity_matrix_randomness
            Γ1[tril_indices] = Γ1_offdiag_bits
            Γ1[triu_indices] = Γ1_offdiag_bits

        special_bits = []

        for i, k, h, j in self.mallows_sampler._parse_randomness(
            randomness.mallows_randomness
        ):
            S[i] = A.pop(k)
            H[i] = h
            n_extra_bits = 2 * (self.n - i) - k - 1 if h else k
            special_bits += list(indexing.index_to_bitstring(n_extra_bits, j))

        Δ_indices, Γ_indices = self._get_Δ_and_Γ_indices(H, S)

        H_indices = tuple(np.flatnonzero(H))
        n_H_bits = np.sum(H, dtype=int)
        n_Δ_bits = len(Δ_indices[0])
        n_Γ_bits = len(Γ_indices[0])

        assert all(Δ_indices[d].shape == (n_Δ_bits,) for d in (0, 1))
        assert len(Δ_indices) == 2
        assert all(Γ_indices[d].shape == (n_Γ_bits,) for d in (0, 1))
        assert len(Γ_indices) == 2

        assert n_H_bits + n_Δ_bits + n_Γ_bits == len(special_bits)
        H_bits = special_bits[:n_H_bits]
        Δ_bits = special_bits[n_H_bits : n_H_bits + n_Δ_bits]
        Γ_bits = special_bits[n_H_bits + n_Δ_bits :]
        if Δ_bits:
            Δ2[Δ_indices] = Δ_bits
        if H_bits:
            Γ2[H_indices, H_indices] = H_bits
        if Γ_bits:
            Γ2[Γ_indices] = Γ_bits
            Γ2[Γ_indices[::-1]] = Γ_bits

        return CliffordSample(
            first_parity_matrix=tuple(tuple(row) for row in Δ1),
            first_phase_matrix=tuple(tuple(row) for row in Γ1),
            hadamards=tuple(H),
            permutation=tuple(S),
            parity_shift=randomness.parity_shift_randomness,
            phase_shift=randomness.phase_shift_randomness,
            second_parity_matrix=tuple(tuple(row) for row in Δ2),
            second_phase_matrix=tuple(tuple(row) for row in Γ2),
        )

    def randomness_iter(self) -> Iterable[CliffordRandomness]:
        binom = comb.binom(self.n)
        for HS, Δ, Γ, X, Z in itertools.product(
            self.mallows_sampler.randomness_iter(),
            itertools.product((0, 1), repeat=binom),
            itertools.product((0, 1), repeat=binom + self.n),
            itertools.product((0, 1), repeat=self.n),
            itertools.product((0, 1), repeat=self.n),
        ):
            yield CliffordRandomness(
                mallows_randomness=HS,
                parity_matrix_randomness=tuple(bool(δ) for δ in Δ),
                phase_matrix_randomness=tuple(bool(γ) for γ in Γ),
                parity_shift_randomness=tuple(bool(x) for x in X),
                phase_shift_randomness=tuple(bool(z) for z in Z),
            )

    def validate_randomness(self, randomness: CliffordRandomness) -> bool:
        if not self.mallows_sampler.validate_randomness(randomness.mallows_randomness):
            return False
        binom = comb.binom(self.n)
        return all(
            (
                testing.is_boolean_tuple_of_len(
                    randomness.parity_matrix_randomness, binom
                ),
                testing.is_boolean_tuple_of_len(
                    randomness.phase_matrix_randomness, binom + self.n
                ),
                testing.is_boolean_tuple_of_len(
                    randomness.parity_shift_randomness, self.n
                ),
                testing.is_boolean_tuple_of_len(
                    randomness.phase_shift_randomness, self.n
                ),
            )
        )

    def name(self) -> str:
        return "CliffordSampler"

    def unique_samples_iter(self) -> Iterable[CliffordSample]:
        binom = comb.binom(self.n)
        Δ1, Δ2 = (np.eye(self.n, dtype=linalg.DTYPE) for _ in "12")
        Γ1, Γ2 = (np.zeros((self.n,) * 2, dtype=linalg.DTYPE) for _ in "12")
        tril_indices = indexing.tril_indices(self.n)
        triu_indices = indexing.triu_indices(self.n)

        for H, S in itertools.product(
            itertools.product((0, 1), repeat=self.n),
            itertools.permutations(range(self.n)),
        ):
            Δ_indices, Γ_indices = self._get_Δ_and_Γ_indices(H, S)
            H_indices = np.flatnonzero(H)
            for H_bits, Δ1_bits, Γ1_bits, Δ2_bits, Γ2_bits in itertools.product(
                itertools.product((0, 1), repeat=len(H_indices)),
                itertools.product((0, 1), repeat=binom),
                itertools.product((0, 1), repeat=binom + self.n),
                itertools.product((0, 1), repeat=len(Δ_indices[0])),
                itertools.product((0, 1), repeat=len(Γ_indices[0])),
            ):
                Γ1[np.diag_indices(self.n)] = Γ1_bits[: self.n]
                assert Γ1[tril_indices].shape == (binom,)
                if self.n > 1:
                    Δ1[tril_indices] = Δ1_bits
                    Γ1[tril_indices] = Γ1_bits[self.n :]
                    Γ1[triu_indices] = Γ1_bits[self.n :]
                    Δ2[tril_indices] = 0
                if Δ2_bits:
                    Δ2[Δ_indices] = Δ2_bits
                Γ2[:, :] = 0
                if len(H_indices):
                    Γ2[H_indices, H_indices] = H_bits
                if Γ2_bits:
                    Γ2[Γ_indices] = Γ2_bits
                    Γ2[Γ_indices[::-1]] = Γ2_bits

                for X, Z in itertools.product(
                    itertools.product((0, 1), repeat=self.n), repeat=2
                ):
                    yield CliffordSample(
                        first_parity_matrix=tuple(tuple(row) for row in Δ1),
                        first_phase_matrix=tuple(tuple(row) for row in Γ1),
                        parity_shift=tuple(bool(x) for x in X),
                        phase_shift=tuple(bool(z) for z in Z),
                        permutation=tuple(S),
                        hadamards=tuple(bool(h) for h in H),
                        second_parity_matrix=tuple(tuple(row) for row in Δ2),
                        second_phase_matrix=tuple(tuple(row) for row in Γ2),
                    )

    def num_unique_samples(self) -> int:
        return comb.num_cliffords(self.n)
