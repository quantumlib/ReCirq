import itertools
import math
from typing import Any, Iterable, Optional, Tuple, cast

import numpy as np
from quaff import comb, indexing, linalg
from quaff.sampling.sampler import SingleParameterSampler

MallowsSample = Tuple[int, ...]
MallowsRandomness = Tuple[int, ...]


class MallowsSampler(SingleParameterSampler):
    def __init__(self, n: int):
        self.n = n

    def randomness_iter(self) -> Iterable[MallowsRandomness]:
        return itertools.product(*(range(k) for k in 2 ** np.arange(self.n, 0, -1) - 1))

    def sample_randomness(
        self, rng: Optional[np.random.Generator] = None
    ) -> MallowsRandomness:
        rng = np.random.default_rng(rng)
        return tuple(rng.integers(0, 2**m - 1) for m in np.arange(self.n, 0, -1))

    def _parse_randomness(self, randomness) -> Iterable[Tuple[int, int, int]]:
        for i, r in enumerate(randomness):
            m = self.n - i
            threshold = 0
            for k in range(1, m + 1):
                threshold += 2 ** (k - 1)
                if r < threshold:
                    break
            yield i, k - 1, threshold - 1 - r

    def randomness_to_sample(self, randomness: MallowsRandomness) -> MallowsSample:
        S = np.zeros(self.n, dtype=int)
        A = list(range(self.n))
        for i, k, _ in self._parse_randomness(randomness):
            S[i] = A.pop(k)
        return tuple(S)

    def randomness_size(self) -> int:
        return math.prod(2**m - 1 for m in range(1, self.n + 1))

    def validate_randomness(self, randomness: MallowsRandomness) -> bool:
        if any(r < 0 for r in randomness):
            return False
        return all(r < 2**m - 1 for r, m in zip(randomness, np.arange(self.n, 0, -1)))

    def sample_multiplicity(self, sample: MallowsRandomness) -> int:
        return 2 ** comb.get_inversion_number(sample)

    def unique_samples_iter(self) -> Iterable[MallowsSample]:
        return itertools.permutations(range(self.n))

    def num_unique_samples(self):
        return math.factorial(self.n)

    def name(self):
        return "MallowsSampler"


InvertibleMatrixSample = Any
InvertibleMatrixRandomness = Tuple[MallowsRandomness, Tuple[bool]]


class InvertibleMatrixSampler(SingleParameterSampler):
    def __init__(self, n: int):
        self.n = n
        self.mallows_sampler = MallowsSampler(n)

    def randomness_iter(self) -> Iterable[InvertibleMatrixRandomness]:
        return cast(
            Iterable[InvertibleMatrixRandomness],
            itertools.product(
                self.mallows_sampler.randomness_iter(),
                itertools.product((False, True), repeat=comb.binom(self.n)),
            ),
        )

    def validate_randomness(self, randomness: InvertibleMatrixRandomness) -> bool:
        if len(randomness) != 2:
            return False
        mallows_randomness, left_bits = randomness
        if not self.mallows_sampler.validate_randomness(mallows_randomness):
            return False
        return all(isinstance(l, bool) for l in left_bits)

    def name(self):
        return "InvertibleMatrixSampler"

    def enumerated_combinations(self):
        return enumerate(itertools.combinations(range(self.n), 2))

    def sample_randomness(
        self, rng: Optional[np.random.Generator]
    ) -> InvertibleMatrixRandomness:
        rng = np.random.default_rng(rng)
        return (
            self.mallows_sampler.sample_randomness(rng),
            rng.integers(0, 2, comb.binom(self.n)),
        )

    def randomness_to_sample(
        self, randomness: InvertibleMatrixRandomness
    ) -> InvertibleMatrixSample:
        mallows_randomness, left_bits = randomness

        S = np.zeros(self.n, dtype=int)
        A = list(range(self.n))
        special_right_bits = []
        for i, k, j in self.mallows_sampler._parse_randomness(mallows_randomness):
            S[i] = A.pop(k)
            special_right_bits += list(indexing.index_to_bitstring(k, j))

        indices = [k for k, (j, i) in self.enumerated_combinations() if S[i] < S[j]]
        right_bits = np.zeros(comb.binom(self.n), dtype=bool)
        assert len(indices) == len(special_right_bits)
        right_bits[indices] = special_right_bits

        L, R = (np.eye(self.n, dtype=linalg.DTYPE) for _ in "LR")
        for k, (j, i) in self.enumerated_combinations():
            L[i, j] = left_bits[k]
            R[i, j] = right_bits[k]
        return tuple(tuple(row) for row in (L[:, S] @ R) % 2)

    def unique_samples_iter(self) -> Iterable[InvertibleMatrixSample]:
        bitstrings = itertools.product((0, 1), repeat=self.n)
        for B in itertools.combinations(bitstrings, self.n):
            _, pivots = linalg.row_reduce(np.array(B, dtype=linalg.DTYPE))
            if len(pivots) != self.n:
                continue
            for BB in itertools.permutations(B):
                yield BB

    def num_unique_samples(self):
        return comb.num_invertible_matrices(self.n)
