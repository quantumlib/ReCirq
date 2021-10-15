# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from scipy.stats import chisquare

import recirq.quantum_chess.bit_utils as u


def print_samples(samples):
    """For debugging only. Prints all the samples as lists of squares."""
    sample_dict = {}
    for sample in samples:
        if sample not in sample_dict:
            sample_dict[sample] = 0
        sample_dict[sample] += 1
    for key in sample_dict:
        print(f"{u.bitboard_to_squares(key)}: {sample_dict[key]}")


def assert_samples_in(b, possibilities):
    samples = b.sample(500)
    assert len(samples) == 500
    all_in = all(sample in possibilities for sample in samples)
    print(possibilities)
    print(set(samples))
    assert all_in, print_samples(samples)
    # make sure each is represented at least once
    for p in possibilities:
        any_in = any(sample == p for sample in samples)
        assert any_in, print_samples(samples)


def assert_sample_distribution(b, probability_map, p_significant=1e-6):
    """Performs a chi-squared test that samples follow an expected distribution.

    probability_map is a map from bitboards to expected probability. An
    assertion is raised if one of the samples is not in the map, or if the
    probability that the samples are at least as different from the expected
    ones as the observed sampless is less than p_significant.
    """
    assert abs(sum(probability_map.values()) - 1) < 1e-9
    samples = b.sample(500)
    assert len(samples) == 500
    counts = defaultdict(int)
    for sample in samples:
        assert sample in probability_map
        counts[sample] += 1
    observed = []
    expected = []
    for position, probability in probability_map.items():
        observed.append(counts[position])
        expected.append(500 * probability)
    p = chisquare(observed, expected).pvalue
    print(observed, expected, "p =", p)
    assert (
        p > p_significant
    ), f"Observed {observed} far from expected {expected} (p = {p})"


def assert_this_or_that(samples, this, that):
    """Asserts all the samples are either equal to this or that,
    and that one of each exists in the samples.
    """
    assert any(sample == this for sample in samples)
    assert any(sample == that for sample in samples)
    assert all(sample == this or sample == that for sample in samples), print_samples(
        samples
    )


def assert_prob_about(probs, that, expected, atol=0.04):
    """Checks that the probability is within atol of the expected value."""
    assert probs[that] > expected - atol
    assert probs[that] < expected + atol


def assert_fifty_fifty(probs, that):
    """Checks that the probability is close to 50%."""
    assert_prob_about(probs, that, 0.5)
