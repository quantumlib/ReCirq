import math
from typing import Any, Iterable, Optional, Tuple

import numpy as np
from recirq.third_party.quaff import linalg, testing
from recirq.third_party.quaff.sampling.sampler import Randomness, Sample, Sampler


class BooleanSampler(Sampler):
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def sample_randomness(self, rng: Optional[np.random.Generator]) -> Randomness:
        rng = np.random.default_rng(rng)
        return linalg.tuplify(rng.integers(0, 2, self.shape))

    def randomness_to_sample(self, randomness: Randomness) -> Sample:
        return randomness

    def randomness_iter(self) -> Iterable[Any]:
        for x in range(2 ** math.prod(self.shape)):
            yield linalg.tuplify(x)

    def validate_randomness(self, randomness: Randomness) -> bool:
        return testing.is_nested_boolean_tuples_of_shape(randomness, self.shape)

    def name(self) -> str:
        return "BooleanArraySampler"

    def parameter_names(self):
        return ("shape",)

    def parameter_values(self) -> Tuple[Any, ...]:
        return (self.shape,)

    def unique_samples_iter(self) -> Iterable[Sample]:
        return self.randomness_iter()

    def num_unique_samples(self) -> int:
        return 2 ** math.prod(self.shape)
