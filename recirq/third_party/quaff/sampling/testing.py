import collections

import numpy as np
import scipy.stats
from recirq.third_party.quaff import json_serialization, random
from recirq.third_party.quaff.testing import assert_equivalent_repr


def assert_sampler_consistent(sampler_cls, *args, test_json=False, **kwargs):
    sampler = sampler_cls(*args, **kwargs)
    R = list(sampler.randomness_iter())
    randomness_size = sampler.randomness_size()
    assert len(R) == randomness_size
    for r in R:
        assert sampler.validate_randomness(r)
    samples = [sampler.randomness_to_sample(r) for r in R]
    unique_samples = list(sampler.unique_samples_iter())
    num_unique_samples = sampler.num_unique_samples()
    assert len(unique_samples) == num_unique_samples
    unique_samples = set(unique_samples)
    assert len(unique_samples) == num_unique_samples
    counter = collections.Counter(samples)
    assert sum(counter.values()) == randomness_size
    for sample, count in counter.items():
        assert count == sampler.sample_multiplicity(sample)
    assert_equivalent_repr(sampler)

    n_samples = 5 * randomness_size
    samples = [sampler.sample(random.RNG) for _ in range(n_samples)]
    if test_json:
        for sample in samples[:10]:
            json_serialization.assert_json_roundtrip_works(sample)
    counter = collections.Counter(samples)
    unique_samples = sorted(counter)
    f_obs = np.array([counter[sample] for sample in unique_samples])
    assert sum(f_obs) == n_samples
    f_exp = np.array([sampler.sample_multiplicity(sample) for sample in unique_samples])
    #   assert sum(f_exp) == randomness_size
    _, p = scipy.stats.chisquare(f_obs, f_exp)


#   assert p < 0.05
# TODO: fix
