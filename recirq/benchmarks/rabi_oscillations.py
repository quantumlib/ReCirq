# Copyright 2022 The Cirq Developers
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

from typing import Any, Optional, Sequence, Tuple
import numpy as np
import sympy

from matplotlib import pyplot as plt

# this is for older systems with matplotlib <3.2 otherwise 3d projections fail
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
import cirq


class RabiResult:
    """Results from a Rabi oscillation experiment. This consists of a set of x-axis
    angles following a sine wave and corresponding measurement probabilities for each.
    """

    def __init__(
        self, rabi_angles: Sequence[float], excited_state_probabilities: Sequence[float]
    ):
        """Initializes RabiResult.

        Args:
            rabi_angles: The rotation angles of the qubit around the x-axis
                of the Bloch sphere.
            excited_state_probabilities: The corresponding probabilities that
                the qubit is in the excited state.
        """
        self._rabi_angles = rabi_angles
        self._excited_state_probs = excited_state_probabilities

    @property
    def data(self) -> Sequence[Tuple[float, float]]:
        """Returns a sequence of tuple pairs with the first item being a Rabi
        angle and the second item being the corresponding excited state
        probability.
        """
        return list(zip(self._rabi_angles, self._excited_state_probs))

    def plot(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
        """Plots excited state probability vs the Rabi angle (angle of rotation
        around the x-axis).

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_ylim([0, 1])
        ax.plot(self._rabi_angles, self._excited_state_probs, "ro-", **plot_kwargs)
        ax.set_xlabel(r"Rabi Angle (Radian)")
        ax.set_ylabel("Excited State Probability")
        if show_plot:
            fig.show()
        return ax


def rabi_oscillations(
    sampler: cirq.Sampler,
    qubit: cirq.Qid,
    max_angle: float = 2 * np.pi,
    *,
    repetitions: int = 1000,
    num_points: int = 200,
) -> RabiResult:
    """Runs a Rabi oscillation experiment.

    Rotates a qubit around the x-axis of the Bloch sphere by a sequence of Rabi
    angles evenly spaced between 0 and max_angle. For each rotation, repeat
    the circuit a number of times and measure the average probability of the
    qubit being in the |1> state.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        max_angle: The final Rabi angle in radians.
        repetitions: The number of repetitions of the circuit for each Rabi
            angle.
        num_points: The number of Rabi angles.

    Returns:
        A RabiResult object that stores and plots the result.
    """
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.X(qubit) ** (theta / np.pi))
    circuit.append(cirq.measure(qubit, key="z"))
    sweep = cirq.study.Linspace(
        key="theta", start=0.0, stop=max_angle, length=num_points
    )
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
    angles = np.linspace(0.0, max_angle, num_points)
    excited_state_probs = np.zeros(num_points)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements["z"])

    return RabiResult(angles, excited_state_probs)
