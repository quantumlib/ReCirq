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
"""Results post-processing and error mitigation."""

from typing import (Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
                    Union)

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress

from recirq.fermi_hubbard.execution import (
    ExperimentRun,
    FermiHubbardExperiment,
    run_experiment
)
from recirq.fermi_hubbard.parameters import FermiHubbardParameters

import cirq


@dataclass
class Rescaling:
    """Parameters for amplitudes rescaling. We assume the following
    approximate relation:

        (<n_j〉_exp − n_avg) / (n_j〉_num − n_avg) ≈ b - a η,

    where:
        - the expectation values〈n_j〉_exp and 〈n_j〉_num are obtained from
          experiments and numerics, respectively.
        - n_avg is the average particle number.

    Attributes:
        slope: Resing slope a.
        intercept: Rescaling intercept b.
    """
    slope: float
    intercept: float


@dataclass(init=False)
class PerSiteQuantity:
    """Measured quantity with per-site values.

    Each inner numpy array on the attributes below is a 2D array where the first
    index denotes a Trotter step count and the second index denotes a fermionic
    site. The arrays are of size η x L, where η is number of Trotter steps
    analysed and L is the number of sites.

    Attributes:
        average: List of average values for each chain.
        std_error: List of standard errors of a mean for each chain.
        std_dev: List of standard deviations for chain.
        values: List of list of non-aggregated results. The outer list iterates
            over chains analysed, the inner list iterates over Trotter steps and
            the first dimension of a numpy array are experiment realizations
            (different configurations).
    """
    average: List[np.ndarray]
    std_error: List[np.ndarray]
    std_dev: Optional[List[np.ndarray]]
    values: Optional[List[List[np.ndarray]]]

    def __init__(self, *,
                 average: Iterable[Iterable[np.ndarray]],
                 std_error: Iterable[Iterable[np.ndarray]],
                 std_dev: Optional[Iterable[Iterable[np.ndarray]]] = None,
                 values: Optional[List[List[np.ndarray]]] = None
                 ) -> None:
        self.average = list(np.array(a) for a in average)
        self.std_error = list(np.array(e) for e in std_error)
        self.std_dev = list(np.array(d) for d in std_dev) if std_dev else None
        self.values = values

    @property
    def chains_count(self) -> int:
        return len(self.average)


@dataclass(init=False)
class AggregatedQuantity:
    """Measured quantity with per-chain values.

    Each inner numpy array on the attributes below is a 1D array indexed by a
    Trotter step count, it is of size η, where η is number of Trotter steps
    analysed.

    Attributes:
        average: List of average values for each chain.
        std_error: List of standard errors of a mean for each chain.
        std_dev: List of standard deviations for chain.
        values: List of list of non-aggregated results. The outer list iterates
            over chains analysed, the inner list iterates over Trotter steps
            and the numpy array lists values for different experiment
            realizations (configurations).
    """

    average: List[np.ndarray]
    std_error: List[np.ndarray]
    std_dev: Optional[List[np.ndarray]]
    values: Optional[List[List[np.ndarray]]]

    def __init__(self, *,
                 average: Iterable[Iterable[float]],
                 std_error: Iterable[Iterable[float]],
                 std_dev: Optional[Iterable[Iterable[float]]] = None,
                 values: Optional[List[List[np.ndarray]]] = None
                 ) -> None:
        self.average = list(np.array(a) for a in average)
        self.std_error = list(np.array(e) for e in std_error)
        self.std_dev = list(np.array(d) for d in std_dev) if std_dev else None
        self.values = values

    @property
    def chains_count(self) -> int:
        return len(self.average)


class InstanceBundle:
    """Bundle of many realizations of the same Fermi-Hubbard problem.

    This class is a holder for post-processed data extracted over many
    realizations (with different qubits assignments assigned by different
    layouts).

    It calculates various quantities with or without amplitudes rescaling. The
    quantities are averaged over all experiments passed on initialization.

    The calculate rescaled quantities (or more precisely the intrinsic
    rescaling), exact numerical simulations needs to be calculated. They are
    calculated only once and cached for later use.
    """

    def __init__(self,
                 experiments: Iterable[FermiHubbardExperiment],
                 steps: Optional[Iterable[int]] = None,
                 rescale_steps: Optional[Iterable[int]] = None,
                 post_select: bool = True,
                 numerics_transform: Optional[
                     Callable[[FermiHubbardParameters], FermiHubbardParameters]
                 ] = None,
                 exact_numerics: bool = False) -> None:
        """Initializes experiment instance bundle.

        The initializer is fast and all the quantities are calculated lazily, on
        demand.

        Args:
            experiments: List of experiments to analyze. They must solve the
                same Fermi-Hubbard problem (have exactly the same Hamiltonian,
                initial state and Trotter step length). The only part which
                might differ is qubits layout, which realizes different
                configurations.
            steps: Trotter steps to analyse. If provided, it must be a subset of
                Trotter steps that occur on experiments list. If not provided,
                the intersection of Trotter steps that occur on experiments
                list will be used.
            rescale_steps: Trotter steps which should be used for rescaling
                technique. If not provided, all steps used for analysis will
                be used. It might be desirable to restrict rescaling steps to
                only the high quality experiments (for example of lower depth).
            post_select: If true, the post selected results are analysed.
            numerics_transform: Optional function that transforms the problem
                parameters before running the numerical simulations (numerical
                simulation are necessary in order to obtain the reference
                amplitudes for rescaling). For example, it might be used to
                compensate for parasitic cphase.
            exact_numerics: If true, indicates that this bundle is result of
                exact numerical simulation and rescaling technique might assume
                perfect amplitudes. This variable is used internally.
        """
        self.experiments = list(experiments)
        self.exact_numerics = exact_numerics
        self._exact_numerics_bundle: Optional['InstanceBundle'] = None
        self._numerics_transform = numerics_transform
        self._post_select = post_select
        self._steps, self._runs, self._parameters = _split_experiments(
            experiments, steps)
        self._rescale_steps = list(rescale_steps) if rescale_steps else None
        self._rescaling = None

    @property
    def representative_parameters(self) -> FermiHubbardParameters:
        """Problem parameters with a default layout."""
        return self._parameters

    @property
    def steps(self) -> List[int]:
        """List of Trotter steps analysed."""
        return self._steps

    @property
    def u(self) -> float:
        """Interaction strength parameter analysed."""
        return self._parameters.hamiltonian.u

    @property
    def dt(self) -> float:
        """Trotter step length analysed."""
        return self._parameters.dt

    @property
    @lru_cache(maxsize=None)
    def quantities(self) -> Dict[str, Callable[[], Union[None,
                                                         PerSiteQuantity,
                                                         AggregatedQuantity]]]:
        quantity_funcs = [
            self.up_down_density,
            self.up_down_position_average,
            self.up_down_position_average_dt,
            self.up_down_spreading,
            self.up_down_spreading_dt,
            self.charge_spin_density,
            self.charge_spin_position_average,
            self.charge_spin_position_average_dt,
            self.charge_spin_spreading,
            self.charge_spin_spreading_dt
        ]
        return {
            **{func.__name__: partial(func, True) for func in quantity_funcs},
            **{f'{func.__name__}_norescale': partial(func, False)
               for func in quantity_funcs},
            **{'scaling': self.scaling,
               'post_selection': self.post_selection}
        }

    @property
    def rescaling(self) -> Optional[Rescaling]:
        """The rescaling value used for analysis.

        This might be set to arbitrary value and override the intrinsic
        rescaling of the underlying data set.
        """
        if self._rescaling is None:
            return self.intrinsic_rescaling
        return self._rescaling

    @rescaling.setter
    def rescaling(self, value: Rescaling) -> None:
        self._rescaling = value
        self.up_down_density.cache_clear()
        self.up_down_position_average.cache_clear()
        self.up_down_position_average_dt.cache_clear()
        self.up_down_spreading.cache_clear()
        self.up_down_spreading_dt.cache_clear()
        self.charge_spin_density.cache_clear()
        self.charge_spin_position_average.cache_clear()
        self.charge_spin_position_average_dt.cache_clear()
        self.charge_spin_spreading.cache_clear()
        self.charge_spin_spreading_dt.cache_clear()

    @property
    @lru_cache(maxsize=None)
    def intrinsic_rescaling(self) -> Optional[Rescaling]:
        """Rescaling for the underlying data set calculated against the exact
        numerical simulations.
        """
        if self.exact_numerics:
            return None
        return _find_rescaling_factors(self.scaling(),
                                       self.steps,
                                       self._rescale_steps)

    @lru_cache(maxsize=None)
    def up_down_density(self, rescaled: bool = True) -> PerSiteQuantity:
        rescaling = self.rescaling if rescaled else None
        return _extract_up_down_densities(self._parameters.up_down_particles,
                                          self._runs,
                                          rescaling,
                                          post_select=self._post_select)

    @lru_cache(maxsize=None)
    def up_down_position_average(self, rescaled: bool = True
                                 ) -> AggregatedQuantity:
        return _extract_position_average(self.up_down_density(rescaled))

    @lru_cache(maxsize=None)
    def up_down_position_average_dt(self, rescaled: bool = True
                                    ) -> AggregatedQuantity:
        return _find_derivative(self.up_down_position_average(rescaled),
                                self.dt)

    @lru_cache(maxsize=None)
    def up_down_spreading(self, rescaled: bool = True) -> AggregatedQuantity:
        return _extract_spreading(self.up_down_density(rescaled))

    @lru_cache(maxsize=None)
    def up_down_spreading_dt(self, rescaled: bool = True) -> AggregatedQuantity:
        return _find_derivative(self.up_down_spreading(rescaled), self.dt)

    @lru_cache(maxsize=None)
    def charge_spin_density(self, rescaled: bool = True) -> PerSiteQuantity:
        rescaling = self.rescaling if rescaled else None
        return _extract_charge_spin_densities(
            self._parameters.up_down_particles,
            self._runs,
            rescaling,
            post_select=self._post_select)

    @lru_cache(maxsize=None)
    def charge_spin_position_average(self, rescaled: bool = True
                                     ) -> AggregatedQuantity:
        return _extract_position_average(self.charge_spin_density(rescaled))

    @lru_cache(maxsize=None)
    def charge_spin_position_average_dt(self, rescaled: bool = True
                                        ) -> AggregatedQuantity:
        return _find_derivative(self.charge_spin_position_average(rescaled),
                                self.dt)

    @lru_cache(maxsize=None)
    def charge_spin_spreading(self, rescaled: bool = True
                              ) -> AggregatedQuantity:
        return _extract_spreading(self.charge_spin_density(rescaled))

    @lru_cache(maxsize=None)
    def charge_spin_spreading_dt(self, rescaled: bool = True
                                 ) -> AggregatedQuantity:
        return _find_derivative(self.charge_spin_spreading(rescaled), self.dt)

    @lru_cache(maxsize=None)
    def scaling(self) -> Optional[AggregatedQuantity]:
        if self.exact_numerics:
            return None
        up_down = self.up_down_density(rescaled=False)
        up_down_numerics = self.exact_numerics_bundle().up_down_density(
            rescaled=False)
        return _find_quantity_scaling(self._parameters.up_down_particles,
                                      up_down,
                                      up_down_numerics)

    @lru_cache(maxsize=None)
    def post_selection(self) -> AggregatedQuantity:
        return _extract_post_selection(self._runs)

    def exact_numerics_bundle(self) -> 'InstanceBundle':
        """Retrieves the exact numerical simulation for an underlying problem.

        It triggers the simulation if neither self.exact_numerics_bundle nor
        self.cache_exact_numerics ran before.

        Returns:
            The instance of InstanceBundle which represents the exact numerical
            simulation results for the underlying problem parameters.
        """
        self.cache_exact_numerics()
        return self._exact_numerics_bundle

    def cache_exact_numerics(
            self,
            keep_raw_result: bool = False,
            threads: int = 4,
            pre_run_func: Optional[Callable[[int, cirq.Circuit], None]] = None,
            post_run_func: Optional[Callable[[int, cirq.Result], None]] = None
    ) -> None:
        """Calculates the exact numerical simulation for an underlying problem.

        Returns:
            The instance of InstanceBundle which represents the exact numerical
            simulation results for the underlying problem parameters.
        """

        if self._exact_numerics_bundle is not None:
            return

        if self.exact_numerics:
            self._exact_numerics_bundle = self

        if self._numerics_transform:
            parameters = self._numerics_transform(self._parameters)
        else:
            parameters = self._parameters

        simulation = run_experiment(parameters,
                                    self.steps,
                                    keep_raw_result=keep_raw_result,
                                    threads=threads,
                                    pre_run_func=pre_run_func,
                                    post_run_func=post_run_func)
        self._exact_numerics_bundle = InstanceBundle((simulation,),
                                                     self.steps,
                                                     exact_numerics=True)


def _extract_up_down_densities(
        up_down_particles: Tuple[int, int],
        runs: Iterable[Iterable[ExperimentRun]],
        rescaling: Optional[Rescaling] = None,
        post_select: bool = True
) -> PerSiteQuantity:

    def up_down_transformation(up_density: np.ndarray,
                               down_density: np.ndarray,
                               ) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(up_density), np.array(down_density)

    return _extract_densities(up_down_particles=up_down_particles,
                              runs=runs,
                              transformation_func=up_down_transformation,
                              rescaling=rescaling,
                              post_select=post_select)


def _extract_charge_spin_densities(
        up_down_particles: Tuple[int, int],
        runs: Iterable[Iterable[ExperimentRun]],
        rescaling: Optional[Rescaling] = None,
        post_select: bool = True
) -> PerSiteQuantity:

    def charge_spin_transformation(up_density: np.ndarray,
                                   down_density: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        up_density = np.array(up_density)
        down_density = np.array(down_density)
        return up_density + down_density, up_density - down_density

    return _extract_densities(up_down_particles=up_down_particles,
                              runs=runs,
                              transformation_func=charge_spin_transformation,
                              rescaling=rescaling,
                              post_select=post_select)


def _extract_densities(
        up_down_particles: Tuple[int, int],
        runs: Iterable[Iterable[ExperimentRun]],
        transformation_func: Callable[[np.ndarray, np.ndarray],
                                      Tuple[np.ndarray, np.ndarray]],
        rescaling: Optional[Rescaling] = None,
        post_select: bool = True
) -> PerSiteQuantity:
    up_particles, down_particles = up_down_particles

    averages = [[], []]
    std_devs = [[], []]
    std_errors = [[], []]
    values = [[], []]

    for run_layouts in runs:
        for chain in range(2):
            values[chain].append([])

        for run in run_layouts:
            if post_select:
                results = run.result_post_selected
            else:
                results = run.result_raw

            up_density = np.array(results.up_density)
            down_density = np.array(results.down_density)

            if rescaling:
                up_density = _rescale_density(up_particles,
                                              run.trotter_steps,
                                              up_density,
                                              rescaling)
                down_density = _rescale_density(down_particles,
                                                run.trotter_steps,
                                                down_density,
                                                rescaling)

            first, second = transformation_func(up_density, down_density)
            values[0][-1].append(first)
            values[1][-1].append(second)

        for chain in range(2):
            values[chain][-1] = np.array(values[chain][-1])
            densities = values[chain][-1]
            averages[chain].append(np.average(densities, axis=0))
            std_devs[chain].append(np.std(densities, axis=0, ddof=1))
            std_errors[chain].append(std_devs[chain][-1] /
                                     np.sqrt(len(densities)))

    return PerSiteQuantity(average=averages,
                           std_dev=std_devs,
                           std_error=std_errors,
                           values=values)


def _rescale_density(particles: int,
                     step: int,
                     density: np.ndarray,
                     rescaling: Rescaling) -> np.ndarray:
    scaling = rescaling.slope * step + rescaling.intercept
    center = particles / len(density)
    return (density - center) / scaling + center


def _extract_position_average(quantity: PerSiteQuantity) -> AggregatedQuantity:

    def position_average(values_step: np.ndarray) -> np.ndarray:
        return sum(value * (index + 1)
                   for index, value in enumerate(values_step.T))

    return _aggregate_quantity(quantity, position_average)


def _extract_spreading(quantity: PerSiteQuantity) -> AggregatedQuantity:

    def spreading(values_step: np.ndarray) -> np.ndarray:
        center = 0.5 * (len(values_step.T) + 1)
        return sum(value * abs(index + 1 - center)
                   for index, value in enumerate(values_step.T))

    return _aggregate_quantity(quantity, spreading)


def _aggregate_quantity(quantity: PerSiteQuantity,
                        aggregate_func: Callable[[np.ndarray], np.ndarray]
                        ) -> AggregatedQuantity:
    averages = []
    std_devs = []
    std_errors = []
    values = []
    for values_steps in quantity.values:
        averages.append([])
        std_devs.append([])
        std_errors.append([])
        values.append([])
        for values_step in values_steps:
            aggregate = aggregate_func(values_step)
            values[-1].append(aggregate)
            averages[-1].append(np.average(aggregate, axis=0))
            std_devs[-1].append(np.std(aggregate, axis=0, ddof=1))
            std_errors[-1].append(std_devs[-1][-1] / np.sqrt(len(aggregate)))

    return AggregatedQuantity(average=averages,
                              std_dev=std_devs,
                              std_error=std_errors,
                              values=values)


def _find_derivative(quantity: AggregatedQuantity,
                     dt: float
                     ) -> AggregatedQuantity:

    def find_derivative(values: Sequence[float]) -> List[float]:
        if len(values) < 3:
            raise ValueError('Derivative not supported for data sets with less '
                             'than three data points.')
        derivative = []
        derivative.append((values[1] - values[0]) / dt)
        for i in range(1, len(values) - 1):
            derivative.append((values[i + 1] - values[i - 1]) / (2.0 * dt))
        derivative.append((values[-1] - values[-2]) / dt)
        return derivative

    def find_error(errors: Sequence[float]) -> List[float]:
        if len(errors) < 3:
            raise ValueError('Derivative not supported for data sets with less '
                             'than three data points.')
        derivative_errors = []
        derivative_errors.append(np.sqrt(
            (errors[0] ** 2 + errors[1] ** 2) / dt ** 2))
        for i in range(1, len(errors) - 1):
            derivative_errors.append(np.sqrt(
                (errors[i + 1] ** 2 + errors[i - 1] ** 2)
                / (2.0 * dt) ** 2))
        derivative_errors.append(np.sqrt(
            (errors[-1] ** 2 + errors[-2] ** 2) / dt ** 2))
        return derivative_errors

    return AggregatedQuantity(
        average=(find_derivative(average) for average in quantity.average),
        std_error=(find_error(error) for error in quantity.std_error)
    )


def _extract_post_selection(runs: Iterable[Iterable[ExperimentRun]]
                            ) -> AggregatedQuantity:
    averages = []
    std_devs = []
    std_errors = []
    values = []

    for run_layouts in runs:
        values.append([])
        for run in run_layouts:
            fraction = (run.result_post_selected.measurements_count /
                        run.result.measurements_count)
            values[-1].append(fraction)

        values[-1] = np.array(values[-1])
        fractions = values[-1]
        averages.append(np.average(fractions))
        std_devs.append(np.std(fractions, ddof=1))
        std_errors.append(std_devs[-1] / np.sqrt(len(fractions)))

    return AggregatedQuantity(average=[averages],
                              std_error=[std_errors],
                              std_dev=[std_devs],
                              values=[values])


def _find_rescaling_factors(scaling: AggregatedQuantity,
                            steps: Iterable[int],
                            rescale_steps: Optional[Iterable[int]] = None
                            ) -> Rescaling:

    # Extract steps and values for rescaling.
    rescale_steps = set(rescale_steps) if rescale_steps else None
    scales, errors, scaling_steps = [], [], []
    for index, step in enumerate(steps):
        if rescale_steps is None or step in rescale_steps:
            scaling_steps.append(step)
            scales.append([scaling.average[chain][index]
                           for chain in range(len(scaling.average))])
            errors.append([scaling.std_error[chain][index]
                           for chain in range(len(scaling.std_error))])
            if rescale_steps:
                rescale_steps.remove(step)

    if rescale_steps:
        raise ValueError(f'Missing Trotter steps {rescale_steps} for rescaling')

    if not scales or not errors:
        raise ValueError('No data points for rescaling')

    scaling_steps = np.array(scaling_steps)
    scales = np.array(scales).T
    errors = np.array(errors).T

    # Find the scaling for each site by aggregating all the chains.
    expected = 0
    inv_variance_sum = 0
    for chain in range(len(scales)):
        inv_variance = 1.0 / errors[chain, :] ** 2
        inv_variance_sum += inv_variance
        expected += scales[chain, :] * inv_variance
    expected /= inv_variance_sum

    # Fit a line to the per-site scaling factors.
    def scales_linear_fit(x):
        scale, offset = x
        fit = scaling_steps * scale + offset
        return expected - fit

    scale, offset = least_squares(scales_linear_fit, np.array([0.0, 1.0])).x

    return Rescaling(scale, offset)


def _find_quantity_scaling(particles: Sequence[int],
                           experiment: PerSiteQuantity,
                           numerics: PerSiteQuantity
                           ) -> AggregatedQuantity:

    assert len(particles) == len(experiment.average) == len(numerics.average), (
        "Incompatible array dimensions for scaling calculation")

    scales = []
    std_errors = []
    for experiment_steps, numerics_steps, chain_particles in zip(
            experiment.average, numerics.average, particles):
        scales.append([])
        std_errors.append([])
        for experiment_density, numerics_density in zip(experiment_steps,
                                                        numerics_steps):
            center = chain_particles / len(experiment_density)
            scale, _, _, _, error = linregress(numerics_density - center,
                                               experiment_density - center)
            scales[-1].append(scale)
            std_errors[-1].append(error)

    return AggregatedQuantity(average=scales, std_error=std_errors)


def _split_experiments(experiments: Iterable[FermiHubbardExperiment],
                       steps: Optional[Iterable[int]] = None
                       ) -> Tuple[List[int],
                                  List[List[ExperimentRun]],
                                  FermiHubbardParameters]:
    """Splits a list of experiments into list of runs for each Trotter step.

    Args:
        experiments: List of experiments to analyze. They must solve the same
            Fermi-Hubbard problem (have exactly the same Hamiltonian, initial
            state and Trotter step length). The only part which might differ is
            qubits layout, which realizes different configurations.
        steps: Trotter steps to analyse. If provided, it must be a subset of
            Trotter steps that occur on experiments list. If not provided, the
            intersection of Trotter steps that occur on experiments list will be
            used.

    Returns:
        Tuple of:
         - List of Trotter steps analysed.
         - For each Trotter step, list of experiment runs from different
           experiments.
         - The representative problem parameters (parameters with a default
           layout).
    """

    runs = defaultdict(lambda: [])
    instance = None
    for experiment in experiments:
        parameters = experiment.parameters.representative_parameters()
        if instance is None:
            instance = parameters
        elif instance != parameters:
            raise ValueError(
                'Incompatible Fermi-Hubbard instances for instance analysis')

        for run in experiment.runs:
            runs[run.trotter_steps].append(run)

    steps = list(sorted(steps if steps else runs.keys()))

    missing = set(steps).difference(runs.keys())
    if missing:
        raise ValueError(f'No experiment to cover Trotter steps {steps}')

    if not all(np.array(steps[1:]) - steps[:-1] == 1):
        raise ValueError(f'Nonconsecutive Trotter steps {steps}')

    return steps, [runs[step] for step in steps], instance


def find_bundles_rescalings(
        bundles: Iterable[InstanceBundle]
) -> Tuple[Tuple[List[InstanceBundle], Rescaling], ...]:
    """Finds a rescaling parameters for each subset of compatible instances.

    Args:
        bundles: Iterable of bundles to extract rescaling values from.

    Returns:
        Tuple of subgroups of bundles with an average rescaling value.
    """

    groups = []
    for bundle in bundles:
        parameters = bundle.representative_parameters
        for group in groups:
            group_parameters = group[0].representative_parameters
            if parameters.equals_for_rescaling(group_parameters):
                group.append(bundle)
                break
        else:
            groups.append([bundle])

    def mean_rescaling(group: List[InstanceBundle]) -> Rescaling:
        slopes, intercepts = [], []
        for instance in group:
            slopes.append(instance.intrinsic_rescaling.slope)
            intercepts.append(instance.intrinsic_rescaling.intercept)
        return Rescaling(np.average(slopes), np.average(intercepts))

    return tuple((group, mean_rescaling(group)) for group in groups)


def apply_rescalings_to_bundles(
        rescalings: Tuple[Tuple[List[InstanceBundle], Rescaling], ...]) -> None:
    """Applies the averaged rescaling to each bundle within the subgroups."""
    for group, rescaling in rescalings:
        for bundle in group:
            bundle.rescaling = rescaling
