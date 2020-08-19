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

import numpy as np

from recirq.optimize.mpg import model_policy_gradient


def sum_of_squares(x):
    return np.sum(x ** 4).item()


def test_model_policy_gradient():
    x0 = np.random.randn(5)
    result = model_policy_gradient(
        sum_of_squares,
        x0,
        learning_rate=1e-1,
        decay_rate=0.96,
        decay_steps=10,
        log_sigma_init=-6.0,
        max_iterations=120,
        batch_size=30,
        radius_coeff=3.0,
        warmup_steps=10,
        known_values=None,
    )

    np.testing.assert_allclose(result.x, np.zeros(len(result.x)), atol=1e-2)
    np.testing.assert_allclose(result.fun, 0, atol=1e-7)
    assert isinstance(result.nfev, int)


def test_model_policy_gradient_with_known_values():
    x0 = np.random.randn(5)
    known_xs = [np.ones(5)]
    known_ys = [10.0]
    _ = model_policy_gradient(
        sum_of_squares,
        x0,
        learning_rate=1e-1,
        decay_rate=0.96,
        decay_steps=10,
        log_sigma_init=-6.0,
        max_iterations=50,
        batch_size=30,
        radius_coeff=3.0,
        warmup_steps=10,
        known_values=(known_xs, known_ys),
    )

    assert len(known_xs) == 1
    assert len(known_ys) == 1


def test_model_policy_gradient_limited_iterations():
    x0 = np.random.randn(10)
    result = model_policy_gradient(
        sum_of_squares,
        x0,
        learning_rate=1e-1,
        decay_rate=0.96,
        decay_steps=10,
        log_sigma_init=-6.0,
        batch_size=30,
        radius_coeff=3.0,
        warmup_steps=10,
        known_values=None,
        max_iterations=15,
    )

    assert isinstance(result.x, np.ndarray)
    assert isinstance(result.fun, float)
    assert result.nit == 15


def test_model_policy_gradient_limited_evaluations():
    x0 = np.random.randn(10)
    result = model_policy_gradient(
        sum_of_squares,
        x0,
        learning_rate=1e-1,
        decay_rate=0.96,
        decay_steps=10,
        log_sigma_init=-6.0,
        batch_size=10,
        radius_coeff=3.0,
        warmup_steps=10,
        known_values=None,
        max_iterations=15,
        max_evaluations=95,
    )

    assert isinstance(result.x, np.ndarray)
    assert isinstance(result.fun, float)
    assert result.nfev == 91


def test_model_policy_gradient_with_random_seed():
    x0 = np.random.randn(5)
    result1 = model_policy_gradient(
        sum_of_squares,
        x0,
        learning_rate=1e-1,
        decay_rate=0.96,
        decay_steps=10,
        log_sigma_init=-6.0,
        max_iterations=50,
        batch_size=30,
        radius_coeff=3.0,
        warmup_steps=10,
        random_state=65536,
    )
    result2 = model_policy_gradient(
        sum_of_squares,
        x0,
        learning_rate=1e-1,
        decay_rate=0.96,
        decay_steps=10,
        log_sigma_init=-6.0,
        max_iterations=50,
        batch_size=30,
        radius_coeff=3.0,
        warmup_steps=10,
        random_state=65536,
    )

    np.testing.assert_equal(result1, result2)
