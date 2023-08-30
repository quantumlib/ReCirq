# Copyright 2023 Google
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

import cirq
import numpy as np
import pytest

from . import experiment as kpz

CYCLES = 1
MU = np.inf
SAMPLER = cirq.Simulator()
rng = np.random.default_rng()


def test_muinf_against_analytics():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    expt = kpz.KPZExperiment(CYCLES, MU, 1, theta, phi)
    res = expt.run_experiment_amplitudes(SAMPLER)
    s = np.sin(theta)
    c = np.cos(theta)
    assert np.isclose(res.mean, 2 * s**2, atol=1e-6)
    assert np.isclose(res.variance, 4 * s**2 * c**2, atol=1e-6)
    assert np.isclose(
        res.skewness, (2 * s**4 - 3 * s**2 + 1) / (c**3 * s), atol=1e-6
    )


TRIALS = 100


def test_randmu_against_analytics():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    s = np.sin(theta)
    c = np.cos(theta)
    mu = rng.random()
    expt = kpz.KPZExperiment(CYCLES, mu, TRIALS, theta, phi)
    res = expt.run_experiment_amplitudes(SAMPLER)

    d_mean = res.jackknife_mean()
    assert np.isclose(res.mean, 2 * s**2 * np.tanh(mu), atol=4 * d_mean)

    d_var = res.jackknife_var()
    var_analytic = 2 * s**2 * (1 + np.cos(2 * theta) * np.tanh(mu) ** 2)
    assert np.isclose(res.variance, var_analytic, atol=4 * d_var)

    d_skw = res.jackknife_skw()
    skw_analytic = (
        2
        * np.sqrt(2)
        * (
            (2 * s**4 * np.tanh(mu) ** 2 + 1)
            * (np.sinh(2 * mu) + np.cosh(2 * mu) + 1) ** 2
            - 3 * (np.sinh(4 * mu) + np.cosh(4 * mu) + 1) * s**2
        )
        * np.tanh(mu)
        / (
            (np.sinh(2 * mu) + np.cosh(2 * mu) + 1) ** 2
            * (np.cos(2 * theta) * np.tanh(mu) ** 2 + 1) ** (3 / 2)
            * s
        )
    )
    assert np.isclose(res.skewness, skw_analytic, atol=4 * d_skw)


def test_mu0_against_analytics():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    s = np.sin(theta)
    c = np.cos(theta)
    mu = 0

    # cycle 1
    cycles = 1
    expt = kpz.KPZExperiment(cycles, mu, TRIALS, theta, phi)
    res = expt.run_experiment_amplitudes(SAMPLER)

    d_kur = res.jackknife_kur()
    kur_analytic = 2 / s**2 - 3
    assert np.isclose(res.kurtosis, kur_analytic, atol=4 * d_kur)

    # cycle 2
    cycles = 2
    expt = kpz.KPZExperiment(cycles, mu, TRIALS, theta, phi)
    res = expt.run_experiment_amplitudes(SAMPLER)

    d_mean = res.jackknife_mean()
    assert np.isclose(res.mean, 0, atol=4 * d_mean)

    d_var = res.jackknife_var()
    var_analytic = s**4 * (1 - np.cos(phi)) + (1 / 8) * (3 + np.cos(phi)) * (
        7 * s**2 + np.sin(3 * theta) ** 2
    )
    assert np.isclose(res.variance, var_analytic, atol=4 * d_var)

    d_skw = res.jackknife_skw()
    assert np.isclose(res.skewness, 0, atol=4 * d_skw)


def test_size_independence_muinf():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    mu = np.inf
    cycles = rng.integers(1, 6)
    expt1 = kpz.KPZExperiment(cycles, mu, 1, theta, phi)
    res1 = expt1.run_experiment_amplitudes(SAMPLER)

    expt2 = kpz.KPZExperiment(cycles, mu, 1, theta, phi, num_qubits=2 * cycles + 2)
    res2 = expt2.run_experiment_amplitudes(SAMPLER)

    assert np.isclose(res1.mean, res2.mean, atol=1e-6)
    assert np.isclose(res1.variance, res2.variance, atol=1e-6)
    assert np.isclose(res1.skewness, res2.skewness, atol=1e-6)
    assert np.isclose(res1.kurtosis, res2.kurtosis, atol=1e-6)


def test_size_independence():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    mu = rng.random()
    cycles = rng.integers(1, 6)
    expt1 = kpz.KPZExperiment(cycles, mu, TRIALS, theta, phi)
    res1 = expt1.run_experiment_amplitudes(SAMPLER)

    expt2 = kpz.KPZExperiment(cycles, mu, TRIALS, theta, phi, num_qubits=2 * cycles + 2)
    res2 = expt2.run_experiment_amplitudes(SAMPLER)

    d_mean = np.sqrt(res1.jackknife_mean() ** 2 + res2.jackknife_mean() ** 2)
    assert np.isclose(res1.mean, res2.mean, atol=4 * d_mean)

    d_var = np.sqrt(res1.jackknife_var() ** 2 + res2.jackknife_var() ** 2)
    assert np.isclose(res1.variance, res2.variance, atol=4 * d_var)

    d_skw = np.sqrt(res1.jackknife_skw() ** 2 + res2.jackknife_skw() ** 2)
    assert np.isclose(res1.skewness, res2.skewness, atol=4 * d_skw)

    d_kur = np.sqrt(res1.jackknife_kur() ** 2 + res2.jackknife_kur() ** 2)
    assert np.isclose(res1.kurtosis, res2.kurtosis, atol=4 * d_kur)


REPS = 1000


def test_bitstring_sampler():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    mu = rng.random()
    cycles = rng.integers(1, 6)
    expt = kpz.KPZExperiment(cycles, mu, TRIALS, theta, phi)
    res1 = expt.run_experiment_amplitudes(SAMPLER)
    res2 = expt.run_experiment(SAMPLER, REPS)

    d_mean = np.sqrt(res1.jackknife_mean() ** 2 + res2.jackknife_mean() ** 2)
    assert np.isclose(res1.mean, res2.mean, atol=4 * d_mean)

    d_var = np.sqrt(res1.jackknife_var() ** 2 + res2.jackknife_var() ** 2)
    assert np.isclose(res1.variance, res2.variance, atol=4 * d_var)

    d_skw = np.sqrt(res1.jackknife_skw() ** 2 + res2.jackknife_skw() ** 2)
    assert np.isclose(res1.skewness, res2.skewness, atol=4 * d_skw)

    d_kur = np.sqrt(res1.jackknife_kur() ** 2 + res2.jackknife_kur() ** 2)
    assert np.isclose(res1.kurtosis, res2.kurtosis, atol=4 * d_kur)


def test_bitstring_sampler_muinf():
    theta = rng.random() * np.pi / 2
    phi = rng.random() * np.pi
    mu = np.inf
    cycles = rng.integers(1, 6)
    expt = kpz.KPZExperiment(cycles, mu, 1, theta, phi)
    res1 = expt.run_experiment_amplitudes(SAMPLER)
    res2 = expt.run_experiment(SAMPLER, REPS)

    d_mean = np.sqrt(res1.jackknife_mean() ** 2 + res2.jackknife_mean() ** 2)
    assert np.isclose(res1.mean, res2.mean, atol=4 * d_mean)

    d_var = np.sqrt(res1.jackknife_var() ** 2 + res2.jackknife_var() ** 2)
    assert np.isclose(res1.variance, res2.variance, atol=4 * d_var)

    d_skw = np.sqrt(res1.jackknife_skw() ** 2 + res2.jackknife_skw() ** 2)
    assert np.isclose(res1.skewness, res2.skewness, atol=4 * d_skw)

    d_kur = np.sqrt(res1.jackknife_kur() ** 2 + res2.jackknife_kur() ** 2)
    assert np.isclose(res1.kurtosis, res2.kurtosis, atol=4 * d_kur)
