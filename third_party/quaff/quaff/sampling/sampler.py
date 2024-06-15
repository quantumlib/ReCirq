import abc
from typing import Any, Iterable, Optional, Tuple

import numpy as np

Randomness = Any
Sample = Any


class Sampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample_randomness(self, rng: Optional[np.random.Generator]) -> Randomness:
        """Uniformly samples randomness."""

    @abc.abstractmethod
    def randomness_to_sample(self, randomness: Randomness) -> Sample:
        """Parses randomness and produce a sample."""

    @abc.abstractmethod
    def randomness_iter(self) -> Iterable[Any]:
        """Iterates over all instantiations of randomness."""

    @abc.abstractmethod
    def validate_randomness(self, randomness: Randomness) -> bool:
        """Returns True if randomness is valid and False otherwise."""

    @abc.abstractmethod
    def name(self) -> str:
        """The name of the sampler."""

    @abc.abstractmethod
    def parameter_names(self) -> Tuple[str, ...]:
        """Returns the names of the parameters specifying an instance of the
        sampler (i.e., the identifiers of the keyword arguments in the
        constructor)."""

    @abc.abstractmethod
    def parameter_values(self) -> Tuple[Any, ...]:
        """Returns the tuple of parameters specifying the instance of the
        sampler (i.e., the values of the keyword arguments in the
        constructor)."""

    @abc.abstractmethod
    def unique_samples_iter(self) -> Iterable[Sample]:
        """Iterates over unique samples."""

    @abc.abstractmethod
    def num_unique_samples(self) -> int:
        """The number of unique samples."""

    def randomness_size(self) -> int:
        """Number of instantiations of randomness."""
        return self.num_unique_samples()

    def sample_multiplicity(self, sample: Sample) -> int:
        """Returns the number of instantiations of randomness that map to the
        given sample."""
        return 1

    def __str__(self):
        arg_str = ", ".join(
            f"{key}={val}"
            for key, val in zip(self.parameter_names(), self.parameter_values())
        )
        return f"{self.name()}({arg_str})"

    def __repr__(self):
        return f"quaff.{self}"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.parameter_values() == other.parameter_values()

    def sample(self, rng: Optional[np.random.Generator] = None) -> Sample:
        """Returns a random sample."""
        randomness = self.sample_randomness(rng)
        return self.randomness_to_sample(randomness)


class SingleParameterSampler(Sampler, metaclass=abc.ABCMeta):
    def __init__(self, n: int):
        self.n = n

    def parameter_names(self):
        return ("n",)

    def parameter_values(self):
        return (self.n,)
