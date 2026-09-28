# Copyright 2026 Google
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

import scipy
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import copy
import dataclasses


def fit_logical_error_per_cycle(
    cycles: np.ndarray, lep: np.ndarray, d_lep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the logical error per cycle.

    See Section III of the SM to Nature 614, 676–681 (2023).

    Args:
        cycles: The cycles for which the lep is measured.
        lep: The logical error probability.
        d_lep: The statistical uncertainty of lep.

    Returns:
        The parameters A and epsilon in the fit and their covariance matrix.
    """
    fit_function = lambda t, amp, epsilon: amp * (1 - 2 * epsilon) ** t
    fidelity = 1 - 2 * lep
    d_fidelity = 2 * d_lep
    fit = scipy.optimize.curve_fit(
        fit_function,
        cycles,
        fidelity,
        sigma=d_fidelity,
        absolute_sigma=True,
        p0=(1.0, 0.01),
    )
    return fit


def get_color(distance: int) -> str:
    """Get the color to use for plotting.

    Meant to reproduce Fig 1 from https://www.nature.com/articles/s41586-024-08449-y.

    Args:
        distance: The code distance.

    Returns:
        The color to use for plotting.

    Raises:
        ValueError: If the code distance is not 3, 5, or 7.
    """
    if distance == 3:
        return "r"
    elif distance == 5:
        return "skyblue"
    elif distance == 7:
        return "b"
    else:
        raise ValueError("Distance must be 3, 5, or 7")


def get_marker(distance: int) -> str | tuple:
    """Get the marker shape to use for plotting.

    Meant to reproduce Fig 1 from https://www.nature.com/articles/s41586-024-08449-y.

    Args:
        distance: The code distance.

    Returns:
        The marker shape to use for plotting.

    Raises:
        ValueError: If the code distance is not 3, 5, or 7.
    """
    if distance == 3:
        return "v"
    elif distance == 5:
        return "p"
    elif distance == 7:
        return (7, 0, 0)
    else:
        raise ValueError("Distance must be 3, 5, or 7")


@dataclasses.dataclass(frozen=True)
class SurfaceCodeParams:
    """Contains parameters that describe a surface code experiment.

    Attributes:
        distance: The code distance.
        shift: Determines which qubits are used.
        observable: Which basis to measure in.
    """

    distance: int
    observable: Literal["H", "V"]
    shift: tuple[int, int]


class LambdaExperimentResults:
    def __init__(
        self, cycles_list: list[int], repetitions: int, num_sweep_bit_choices: int
    ):
        self.lep_all = []
        self.cycles_list = cycles_list
        self.num_sweep_bit_choices = num_sweep_bit_choices
        self.repetitions = repetitions
        self.params_all = []
        self.shuffled_cycles_all = []
        self.avg_lep_by_distance = {}
        self.d_avg_lep_by_distance = {}
        self.fitted_ler = {}
        self.d_fitted_ler = {}
        self.fitted_scale = {}
        self.d_fitted_scale = {}

    def add_result(
        self, params: SurfaceCodeParams, lep: np.ndarray, shuffled_cycles: np.ndarray
    ):
        self.params_all.append(params)
        self.shuffled_cycles_all.append(shuffled_cycles)

        # unshuffle and reshape lep:
        order = np.argsort(shuffled_cycles)
        self.lep_all.append(
            lep[order].reshape(len(self.cycles_list), self.num_sweep_bit_choices)
        )

    def average_instances(self):
        distances = sorted({params.distance for params in self.params_all})
        lep_by_distance = {distance: [] for distance in distances}
        for lep, params in zip(self.lep_all, self.params_all):
            lep_by_distance[params.distance].append(lep)
        for distance in distances:
            self.avg_lep_by_distance[distance] = np.mean(
                np.mean(lep_by_distance[distance], axis=0), axis=1
            )  # average over params and then sweep bits
            # the following is approximate and may underestimate the uncertainty
            # from sampling params if few params are sampled
            self.d_avg_lep_by_distance[distance] = np.std(
                np.array(lep_by_distance[distance])
                .transpose(0, 2, 1)
                .reshape(-1, len(self.cycles_list)),
                ddof=1,
                axis=0,
            ) / np.sqrt(self.num_sweep_bit_choices * len(lep_by_distance[distance]))

    def fit_exponential(self):
        self.average_instances()
        for distance, lep in self.avg_lep_by_distance.items():
            d_lep = copy.deepcopy(self.d_avg_lep_by_distance[distance])
            if np.mean(d_lep) < 1e-5:
                d_lep += 1e-5  # to fit noiseless data
            popt, cov = fit_logical_error_per_cycle(self.cycles_list, lep, d_lep)
            self.fitted_scale[distance] = popt[0]
            self.d_fitted_scale[distance] = np.sqrt(cov[0, 0])
            self.fitted_ler[distance] = popt[1]
            self.d_fitted_ler[distance] = np.sqrt(cov[1, 1])

    def plot(self, ax: plt.Axes) -> plt.Axes:
        self.fit_exponential()
        included_distances = set()
        for params, lep in zip(self.params_all, self.lep_all):
            distance = params.distance
            marker = get_marker(distance)
            color = get_color(distance)
            for sweep_idx in range(self.num_sweep_bit_choices):
                ax.plot(
                    self.cycles_list,
                    lep[:, sweep_idx],
                    marker=marker,
                    color=color,
                    linestyle="none",
                    alpha=0.3,
                    label=(
                        f"$d = {params.distance}$ (individual)"
                        if distance not in included_distances
                        else None
                    ),
                )
                included_distances.add(distance)

        for distance, avg_lep in self.avg_lep_by_distance.items():
            marker = get_marker(distance)
            color = get_color(distance)
            d_avg_lep = self.d_avg_lep_by_distance[distance]
            ax.errorbar(
                self.cycles_list,
                avg_lep,
                d_avg_lep,
                marker=marker,
                color=color,
                linestyle="none",
                capsize=3,
                label=f"$d = {distance}$ (mean)",
                mec="k",
                ecolor="k",
                zorder=100,
            )
            t = np.linspace(0, 250, 100)
            ax.plot(
                t,
                (
                    1
                    - self.fitted_scale[distance]
                    * (1 - 2 * self.fitted_ler[distance]) ** t
                )
                / 2,
                color=color,
                label=f"Fit, LER={self.fitted_ler[distance]*100:.3f}% ± {self.d_fitted_ler[distance]*100:.3f}%",
            )

        if 3 in self.fitted_ler and 5 in self.fitted_ler:
            ax.text(
                175,
                0.5,
                f"$\\Lambda_{{35}} = {self.fitted_ler[3]/self.fitted_ler[5]:.2f} \\pm {np.sqrt( (self.d_fitted_ler[3]/self.fitted_ler[5])**2 + (self.fitted_ler[3]*self.d_fitted_ler[5]/self.fitted_ler[5]**2)**2 ):.2f}$",
                va="top",
            )
        if 5 in self.fitted_ler and 7 in self.fitted_ler:
            ax.text(
                175,
                0.5,
                f"\n$\\Lambda_{{57}} = {self.fitted_ler[5]/self.fitted_ler[7]:.2f} \\pm {np.sqrt( (self.d_fitted_ler[5]/self.fitted_ler[7])**2 + (self.fitted_ler[5]*self.d_fitted_ler[7]/self.fitted_ler[7]**2)**2 ):.2f}$",
                va="top",
            )

        ax.set_xlim(-1, 255)
        ax.set_ylim(-0.01, 0.55)
        ax.set_xlabel("Quantum error correction cycle, $t$")
        ax.set_ylabel("Logical error probability $p_L$")
        ax.legend(frameon=False, labelcolor="linecolor", loc="upper left")
        ax.tick_params(direction="in", top=True, right=True)

        return ax
