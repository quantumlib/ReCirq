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
"""Module for simulating 1D Floquet XXZ dynamics and measuring the transferred magnetization.

This module allows you to perform a numerical version of the experiment detailed in
[arXiv:2306.09333](https://arxiv.org/abs/2306.09333). In particular, it is conjectured that the
late-time dynamics of the 1D Heisenberg spin chain (or the corresponding Floquet system) are
described by the Kardar-Parisi-Zhang (KPZ) universality class. By studying higher moments of
the transferred magnetization, one can see disagreement from the KPZ predictions, at least at
times accessible to us. To measure these moments, we initialize a 1D chain of qubits in product
states sampled from Eq. 3 of [arXiv:2306.09333](https://arxiv.org/abs/2306.09333). Then we apply
alternating layers of fSim gates to implement the Floquet dynamics. Finally, we measure the qubits
and check how many excitations crossed the center of the chain. Twice this quantity is the
transferred magnetization.

The class `KPZExperiment` sets up this experiment for a particular number, $t$, of cycles of fSim
gates, where a cycle is depicted in Figure 1 of [arXiv:2306.09333](https://arxiv.org/abs/2306.09333).
By default, the simulation uses the minimum number of qubits, $2t$. You can then run the
experiment using either the `run_experiment_amplitudes()` method or the `run_experiment()` method.
The former requires you to input a Cirq sampler that supports statevector simulations, whereas the
latter requires only one that can sample bitstrings, closer to what is done in the experiment. A
`KPZExperimentResultsFromAmplitudes` or `KPZExperimentResults` object is returned, from which the
probability distribution of the transferred magnetization can be seen, as well as its first
four moments. The statistical uncertainties of the moments can be computed using the methods
`jackknife_mean()`, `jackknife_variance()`, `jackknife_skew()`, and `jackknife_kurtosis()`.

The `run_experiment()` method differs from what is done on hardware in several important ways.
It does not include any of the post-selection that we do as part of our error mitigation
(since in this tutorial, it is run on a noiseless simulator). Further, in the experiment,
we use 46 qubits for all cycles instead of $2t$ qubits, which is beyond brute-force classical
simulation. On hardware, we also use the same initial bitstrings across cycle numbers, whereas
here they are chosen independently.

"""

from typing import Iterator, List, Union, Optional

import cirq
import numpy as np
import warnings
from tqdm import tqdm
import scipy.stats as sstats
import matplotlib.pyplot as plt

# initialize a random number generator
rng = np.random.default_rng()


def _dec_to_binary_right(d: Union[np.ndarray, int], n: int) -> Union[np.ndarray, int]:
    i = np.arange(n // 2)
    return (
        np.floor(np.outer(d, 1 / 2**i)) - np.floor(np.outer(d, 1 / 2 ** (i + 1))) * 2
    ).astype("int")


class KPZExperimentResultsFromAmplitudes:
    """A class for processing and storing numerical KPZ results.

    An object of this type is returned by `KPZExperiment.run_experiment_amplitudes()`,
    which uses statevector simulations to obtain the probabilities, `prob_right[trial, nR]`, of having
    `nR` excitations on the right side of the chain, given the initial state `initial_states[trial]`.
    The probabilities and initial bitstrings are used to obtain the probability distribution of
    the transferred magnetization, `transferred_magnetization_probs`, which can be plotted with the
    `plot_histogram()` method and is also used to compute the first four moments.
    """

    def __init__(self, prob_right: np.ndarray, initial_states: np.ndarray):
        """
        Args:
            `prob_right`: `prob_right[trial, num_right]` is the probability of measuring num_right excitations
                on the right side of the chain, given the initial state `initial_states[trial]`.
            `initial_states`: An array of the initial bitstrings used in the experiment.
        """

        num_trials, n = initial_states.shape
        num_right = np.sum(initial_states[:, n // 2 :], 1)
        self.num_initial_states = num_trials
        self.num_right_initial = num_right

        self.transferred_magnetization_vals = np.arange(-n // 2, n // 2 + 1) * 2
        transferred_magnetization_probs = np.zeros((num_trials, n + 1))
        for trial in range(num_trials):
            transferred_magnetization_probs[
                trial, (n // 2 - num_right[trial]) : (n - num_right[trial] + 1)
            ] = prob_right[trial, :]
        self.transferred_magnetization_probs_all = transferred_magnetization_probs
        self.transferred_magnetization_probs = np.mean(
            transferred_magnetization_probs, 0
        )
        self.mean = self._mean()
        self.variance = self._variance()
        self.skewness = self._skewness()
        self.kurtosis = self._kurtosis()

    def _mean(self) -> float:
        return (
            self.transferred_magnetization_probs @ self.transferred_magnetization_vals
        )

    def _variance(self) -> float:
        return (
            self.transferred_magnetization_probs
            @ (self.transferred_magnetization_vals - self.mean) ** 2
        )

    def _skewness(self) -> float:
        return (
            self.transferred_magnetization_probs
            @ (self.transferred_magnetization_vals - self.mean) ** 3
            / self.variance ** (3 / 2)
        )

    def _kurtosis(self) -> float:
        return (
            self.transferred_magnetization_probs
            @ (self.transferred_magnetization_vals - self.mean) ** 4
            / self.variance**2
            - 3
        )

    def _mean_excluding_i(self, i: int) -> float:
        p = np.mean(
            np.delete(self.transferred_magnetization_probs_all, i, axis=0), axis=0
        )
        return p @ self.transferred_magnetization_vals

    def _variance_excluding_i(self, i: int) -> float:
        p = np.mean(
            np.delete(self.transferred_magnetization_probs_all, i, axis=0), axis=0
        )
        mean_i = p @ self.transferred_magnetization_vals
        return p @ (self.transferred_magnetization_vals - mean_i) ** 2

    def _skew_excluding_i(self, i: int) -> float:
        p = np.mean(
            np.delete(self.transferred_magnetization_probs_all, i, axis=0), axis=0
        )
        mean_i = p @ self.transferred_magnetization_vals
        variance_i = p @ (self.transferred_magnetization_vals - mean_i) ** 2
        return (
            p
            @ (self.transferred_magnetization_vals - mean_i) ** 3
            / variance_i ** (3 / 2)
        )

    def _kurtosis_excluding_i(self, i: int) -> float:
        p = np.mean(
            np.delete(self.transferred_magnetization_probs_all, i, axis=0), axis=0
        )
        mean_i = p @ self.transferred_magnetization_vals
        variance_i = p @ (self.transferred_magnetization_vals - mean_i) ** 2
        return (
            p @ (self.transferred_magnetization_vals - mean_i) ** 4 / variance_i**2
            - 3
        )

    def jackknife_mean(self) -> float:
        """Compute the statistical uncertainty of the mean using the remove-one jackknife.
        If there is only one initial state (for example if $\mu = \infty$), zero uncertainty
        is returned.
        """
        if self.num_initial_states == 1:
            return 0
        mean_i = [self._mean_excluding_i(i) for i in range(self.num_initial_states)]
        return np.std(mean_i) * np.sqrt(self.num_initial_states - 1)

    def jackknife_variance(self) -> float:
        """Compute the statistical uncertainty of the variance using the remove-one jackknife.
        If there is only one initial state (for example if $\mu = \infty$), zero uncertainty
        is returned.
        """
        if self.num_initial_states == 1:
            return 0
        variance_i = [
            self._variance_excluding_i(i) for i in range(self.num_initial_states)
        ]
        return np.std(variance_i) * np.sqrt(self.num_initial_states - 1)

    def jackknife_skew(self) -> float:
        """Compute the statistical uncertainty of the skewness using the remove-one jackknife.
        If there is only one initial state (for example if $\mu = \infty$), zero uncertainty
        is returned.
        """
        if self.num_initial_states == 1:
            return 0
        skew_i = [self._skew_excluding_i(i) for i in range(self.num_initial_states)]
        return np.std(skew_i) * np.sqrt(self.num_initial_states - 1)

    def jackknife_kurtosis(self) -> float:
        """Compute the statistical uncertainty of the kurtosis using the remove-one jackknife.
        If there is only one initial state (for example if $\mu = \infty$), zero uncertainty
        is returned.
        """
        if self.num_initial_states == 1:
            return 0
        kurtosis_i = [
            self._kurtosis_excluding_i(i) for i in range(self.num_initial_states)
        ]
        return np.std(kurtosis_i) * np.sqrt(self.num_initial_states - 1)

    def plot_histogram(self, ax: Optional[Union[None, plt.Axes]] = None) -> plt.Axes:
        """Plot a histogram of transferred magnetization.
        Args:
            ax: Optional. A matplotlib axes on which to draw the histogram.

        Returns:
            A matplotlib axes on which the histogram is drawn.

        """
        if not ax:
            fig, ax = plt.subplots(facecolor="white", dpi=200)
        bins = (
            np.append(
                self.transferred_magnetization_vals // 2,
                self.transferred_magnetization_vals[-1] // 2 + 1,
            )
            - 0.5
        )
        ax.hist(
            self.transferred_magnetization_vals // 2,
            weights=self.transferred_magnetization_probs,
            bins=bins,
            edgecolor="k",
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlabel("Number of 1s that crossed center, $\mathcal{M}/2$")
        ax.set_ylabel("Probability")
        return ax


class KPZExperimentResults:
    """A class for processing and storing KPZ experiment results.

    An object of this type is returned by `KPZExperiment.run_experiment()`, which uses
    the `run()` method of the Cirq sampler to sample final bitstrings given the initial
    bitstrings specified by the initial_states array. The outputs of `sampler.run()` are
    inputted as `raw_results`. Pooling these results together, the transferred
    magnetization is computed. Its histogram can be visualized using the `plot_histogram()`
    method. The first four moments can also be computed, as well as their statistical
    uncertainties.
    """

    def __init__(
        self,
        raw_results: List[List[cirq.study.result.ResultDict]],
        initial_states: np.ndarray,
    ):
        num_trials, n = initial_states.shape
        num_right = np.sum(initial_states[:, n // 2 :], 1)
        self.num_initial_states = num_trials
        self.num_right_initial = num_right

        self.bitstrs = np.array([res.measurements["m"] for res in raw_results])
        self.num_right_final = np.sum(self.bitstrs[:, :, n // 2 :], 2)

        self.transferred_magnetization = self._transferred_magnetization()
        self.mean = self._mean()
        self.variance = self._variance()
        self.skewness = self._skewness()
        self.kurtosis = self._kurtosis()

    def _transferred_magnetization(self) -> np.ndarray:
        final = self.num_right_final
        num_trials, num_reps = final.shape
        initial = np.outer(self.num_right_initial, np.ones(num_reps, dtype=int))
        return 2 * (final - initial)

    def _mean(self) -> float:
        return np.mean(self.transferred_magnetization.flatten())

    def _variance(self) -> float:
        return np.var(self.transferred_magnetization.flatten())

    def _skewness(self) -> float:
        return sstats.skew(self.transferred_magnetization.flatten())

    def _kurtosis(self) -> float:
        return sstats.kurtosis(self.transferred_magnetization.flatten(), fisher=True)

    def _mean_excluding_i(self, i: int, axis: Optional[int] = 0) -> float:
        tm = np.delete(self.transferred_magnetization, i, axis=axis)
        return np.mean(tm.flatten())

    def _variance_excluding_i(self, i: int, axis: Optional[int] = 0) -> float:
        tm = np.delete(self.transferred_magnetization, i, axis=axis)
        return np.var(tm.flatten())

    def _skew_excluding_i(self, i: int, axis: Optional[int] = 0) -> float:
        tm = np.delete(self.transferred_magnetization, i, axis=axis)
        return sstats.skew(tm.flatten())

    def _kurtosis_excluding_i(self, i: int, axis: Optional[int] = 0) -> float:
        tm = np.delete(self.transferred_magnetization, i, axis=axis)
        return sstats.kurtosis(tm.flatten(), fisher=True)

    def jackknife_mean(self) -> float:
        """Compute the statistical uncertainty of the mean using the remove-one jackknife.
        In the case that there is only one initial state, use the standard deviation of
        the measured transferred magnetization to estimate the uncertainty instead.
        """
        if self.num_initial_states == 1:
            tm = self.transferred_magnetization.flatten()
            return np.std(tm) / np.sqrt(len(tm))
        mean_i = [self._mean_excluding_i(i) for i in range(self.num_initial_states)]
        return np.std(mean_i) * np.sqrt(self.num_initial_states - 1)

    def jackknife_variance(self) -> float:
        """Compute the statistical uncertainty of the variance using the remove-one jackknife.
        One initial state is removed, and the variation depending on which state is removed
        is used to estimate the uncertainty. In the case that there is only one initial state,
        a repetition is removed instead.
        """
        if self.num_initial_states == 1:
            axis = 1
            tot = self.transferred_magnetization.size
        else:
            axis = 0
            tot = self.num_initial_states
        variance_i = [self._variance_excluding_i(i, axis=axis) for i in range(tot)]
        return np.std(variance_i) * np.sqrt(tot - 1)

    def jackknife_skew(self) -> float:
        """Compute the statistical uncertainty of the skewness using the remove-one jackknife.
        One initial state is removed, and the variation depending on which state is removed
        is used to estimate the uncertainty. In the case that there is only one initial state,
        a repetition is removed instead.
        """
        if self.num_initial_states == 1:
            axis = 1
            tot = self.transferred_magnetization.size
        else:
            axis = 0
            tot = self.num_initial_states
        skew_i = [self._skew_excluding_i(i, axis=axis) for i in range(tot)]
        return np.std(skew_i) * np.sqrt(tot - 1)

    def jackknife_kurtosis(self) -> float:
        """Compute the statistical uncertainty of the kurtosis using the remove-one jackknife.
        One initial state is removed, and the variation depending on which state is removed
        is used to estimate the uncertainty. In the case that there is only one initial state,
        a repetition is removed instead.
        """
        if self.num_initial_states == 1:
            axis = 1
            tot = self.transferred_magnetization.size
        else:
            axis = 0
            tot = self.num_initial_states
        kurtosis_i = [self._kurtosis_excluding_i(i, axis=axis) for i in range(tot)]
        return np.std(kurtosis_i) * np.sqrt(tot - 1)

    def plot_histogram(self, ax: Optional[Union[None, plt.Axes]] = None) -> plt.Axes:
        """
        Plot a histogram of transferred magnetization.
        Args:
            ax: Optional. A matplotlib axes on which to draw the histogram.

        Returns:
            A matplotlib axes on which the histogram is drawn.

        """
        if not ax:
            fig, ax = plt.subplots(facecolor="white", dpi=200)
        num_crossed = self.transferred_magnetization // 2
        lower = min(num_crossed.flatten())
        upper = max(num_crossed.flatten())
        ax.hist(
            num_crossed.flatten(),
            bins=np.arange(lower - 0.5, upper + 1.5),
            density=True,
            edgecolor="k",
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlabel("Number of 1s that crossed center, $\mathcal{M}/2$")
        ax.set_ylabel("Probability")
        return ax


class KPZExperiment:
    """A class for running/simulating the KPZ experiment.

    This class implements 1D Floquet XXZ dynamics, realized as alternating layers of fSim
    gates. The initial states, parameterized by mu, interpolate between an
    infinite-temperature/maximally mixed state at $\mu=0$ and a pure domain wall
    state at $\mu=\infty$. (See Eq. 3 of [arXiv:2306.09333](https://arxiv.org/pdf/2306.09333.pdf).) The transferred
    magnetization (the number of 1s that cross the center) is measured and
    its moments are computed.

    The fSim gates are parameterized by `theta` and `phi`. The isotropic Heisenberg point, at which
    the KPZ conjecture applies, corresponds to `phi = 2*theta`.

    The transferred magnetization is independent of system size up to $N/2$ cycles, where $N$
    is the number of qubits. Therefore, in this class, we use $2t$ qubits to simulate cycle $t$.
    In the experiment, we use 46 qubits to simulate cycles 0-23.

    """

    def __init__(
        self,
        num_cycles: int,
        mu: float,
        num_init_states: int,
        theta: float,
        phi: float,
        num_qubits: Optional[Union[None, int]] = None,
    ):
        """

        Args:
            `num_cycles`: The number of cycles to simulate.
            `mu`: A parameter that controls the initial state.
            `num_init_states`: The number of initial bitstrings to sample.
            `theta`: fSim swap angle in radians.
            `phi`: fSim cphase angle in radians.
            `num_qubits`: The number of qubits to use. Defaults to `2*num_cycles`. The actual
                experiment uses 46.
        """

        if mu == np.inf and num_init_states > 1:
            warnings.warn(
                "When mu=inf, there is only 1 initial state. Setting num_init_states = 1"
            )
            num_init_states = 1

        self.num_cycles = num_cycles
        self.mu = mu
        self.num_init_states = num_init_states
        self.num_qubits = 2 * num_cycles if num_qubits == None else num_qubits
        if self.num_qubits == 0:
            self.num_qubits = 2

        self.initial_states = self._generate_initial_states()
        self.theta = theta
        self.phi = phi
        self.circuits = self._generate_all_circuits()

    def _generate_initial_states(self) -> np.ndarray:
        """Generate the initial bitstrings."""

        mu = self.mu
        n = self.num_qubits
        nTrials = self.num_init_states
        if mu == np.inf:
            bitstrs = np.array(
                [np.append(np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int))]
            )
        elif mu < np.inf:
            p = np.exp(mu) / (np.exp(mu) + np.exp(-mu))
            bitstrs_L = rng.choice(2, p=[1 - p, p], size=(nTrials, n // 2))
            bitstrs_R = rng.choice(2, p=[p, 1 - p], size=(nTrials, n // 2))
            bitstrs = np.append(bitstrs_L, bitstrs_R, axis=1)
        return bitstrs

    def _generate_cycle(self) -> cirq.Circuit:
        """Generate the Cirq circuit for one cycle."""

        n = self.num_qubits
        theta = self.theta
        phi = self.phi
        qubits = cirq.LineQubit.range(n)

        qc = cirq.Circuit(
            cirq.FSimGate(theta, phi)(*qubits[q : q + 2]) for q in range(0, n - 1, 2)
        )
        qc += cirq.Circuit(
            cirq.FSimGate(theta, phi)(*qubits[q : q + 2]) for q in range(1, n - 1, 2)
        )

        return qc

    def _generate_circuit(self, locs: Iterator[int]) -> cirq.Circuit:
        """Generate a single circuit.

        Args:
            locs: Locations of 1 in the initial bitstring.

        Returns:
            A Cirq circuit that implements the Floquet dynamics for the specified initial state.
        """

        n = self.num_qubits
        num_cycles = self.num_cycles
        qubits = cirq.LineQubit.range(n)

        qc = cirq.Circuit(cirq.X(qubits[q]) for q in locs)
        if num_cycles > 0:
            cycle = self._generate_cycle()
            qc += cycle * num_cycles
        qc.append(cirq.measure(*qubits, key="m"))
        return qc

    def _generate_all_circuits(self) -> List[cirq.Circuit]:
        """Generate Cirq circuits for all of the initial states."""

        locs_all = [np.nonzero(bitstr)[0] for bitstr in self.initial_states]
        return [self._generate_circuit(locs) for locs in locs_all]

    def run_experiment(self, sampler: cirq.Sampler, reps: int) -> KPZExperimentResults:
        """Run the experiment using the provided Cirq sampler.

        Args:
            `sampler`: The cirq sampler to use for the simulation.
            `reps`: The number of bitstrings to sample per initial state.

        Returns:
            A KPZExperimentResults object containing the measured bitstrings and histogram of
            transferred magnetization, as well as the extracted moments.

        """

        result = [
            sampler.run(circuit, repetitions=reps)
            for circuit in tqdm(self.circuits, total=self.num_init_states)
        ]
        return KPZExperimentResults(result, self.initial_states)

    def run_experiment_amplitudes(
        self, sampler: cirq.SimulatesAmplitudes
    ) -> KPZExperimentResultsFromAmplitudes:
        """Run the experiment using the provided Cirq sampler. Computes amplitudes instead of sampling bitstrings.

        Args:
            `sampler`: The cirq sampler to use for the simulation.

        Returns:
            A `KPZExperimentResultsFromAmplitudes` object containing the measured bitstrings and histogram of
            transferred magnetization, as well as the extracted moments.

        """

        all_states = np.arange(2**self.num_qubits)
        binary_states = _dec_to_binary_right(all_states, self.num_qubits)
        num_right = np.sum(binary_states, 1)
        del binary_states  # this is exponentially large; delete to save memory
        prob_right = np.zeros((self.num_init_states, self.num_qubits // 2 + 1))
        for idx, (qc, initial_bitstr) in tqdm(
            enumerate(zip(self.circuits, self.initial_states)),
            total=self.num_init_states,
        ):
            num_right_initial = np.sum(initial_bitstr)
            probs = (
                np.abs(
                    sampler.simulate(
                        cirq.drop_terminal_measurements(qc)
                    ).final_state_vector
                )
                ** 2
            )
            for num_right_val in np.arange(
                min(self.num_qubits // 2 + 1, num_right_initial + 1)
            ):
                prob_right[idx, num_right_val] = probs @ (num_right == num_right_val)

        return KPZExperimentResultsFromAmplitudes(prob_right, self.initial_states)
