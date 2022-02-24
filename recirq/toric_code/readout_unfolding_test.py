# Copyright 2022 Google
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
import numpy as np
import pytest

from . import readout_unfolding as ru

DISTRIBUTIONS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0.1, 0.9, 0, 0],
    [0, 0, 0.1, 0.9],
    [0.25, 0.25, 0.25, 0.25],
    [0.1, 0.2, 0.3, 0.4],
]


@pytest.mark.parametrize("use_uniform_prior", [False, True])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_initial_guess(use_uniform_prior, distribution):
    guess = ru._get_initial_guess(distribution, use_uniform_prior)
    if use_uniform_prior:
        assert np.allclose(guess, [0.25, 0.25, 0.25, 0.25])
    else:
        assert np.allclose(guess, distribution)


@pytest.mark.parametrize("use_uniform_prior", [False, True])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_correct_with_identity_matrix(distribution, use_uniform_prior):
    distribution = np.array(distribution)
    corrected = ru.correct_readout_distribution(
        np.eye(4), distribution, use_uniform_prior=use_uniform_prior
    )
    assert np.allclose(corrected, distribution)


@pytest.mark.parametrize("use_uniform_prior", [False, True])
@pytest.mark.parametrize("ideal", DISTRIBUTIONS)
def test_error_on_one_qubit(use_uniform_prior, ideal):
    ideal = np.array(ideal)

    # Use order 00, 10, 01, 11 and put error on the right qubit
    error_0 = 0.05
    error_1 = 0.1
    readout_matrix = np.array(
        [
            [1 - error_0, 0, error_0, 0],  # prepare |00)
            [0, 1 - error_0, 0, error_0],  # prepare |10)
            [error_1, 0, 1 - error_1, 0],  # prepare |01)
            [0, error_1, 0, 1 - error_1],  # prepare |11)
        ]
    )
    measured = readout_matrix.transpose() @ ideal
    corrected = ru.correct_readout_distribution(
        readout_matrix, measured, use_uniform_prior=use_uniform_prior
    )
    assert np.allclose(corrected, ideal, atol=0.01)


@pytest.mark.parametrize("use_uniform_prior", [False, True])
@pytest.mark.parametrize("ideal", DISTRIBUTIONS)
def test_correlated_error(use_uniform_prior, ideal):
    ideal = np.array(ideal)
    readout_matrix = np.array(
        [
            [0.95, 0.02, 0.02, 0.01],  # prepare |00)
            [0.05, 0.90, 0.03, 0.02],  # prepare |10)
            [0.05, 0.03, 0.90, 0.02],  # prepare |01)
            [0.15, 0.05, 0.05, 0.75],  # prepare |11)
        ]
    )
    measured = readout_matrix.transpose() @ ideal
    corrected = ru.correct_readout_distribution(
        readout_matrix, measured, use_uniform_prior=use_uniform_prior
    )
    assert np.allclose(corrected, ideal, atol=0.02)
