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
"""Routines and data types for experiment execution and life cycle."""

from dataclasses import dataclass
from typing import (
    Callable, Dict, IO, Iterable, Optional, Tuple, Type, Union, cast)

import cirq
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import pathlib
import time

from recirq.fermi_hubbard.circuits import create_line_circuits, create_zigzag_circuits
from recirq.fermi_hubbard.layouts import (
    LineLayout,
    ZigZagLayout
)
from recirq.fermi_hubbard.parameters import FermiHubbardParameters


@dataclass(init=False)
class ExperimentResult:
    """Accumulated results of a single Fermi-Hubbard circuit run.

    This container is used to store raw probabilities for each qubit (fermionic
    number densities), with or without post-selection applied.
    """
    up_density: Tuple[float]
    down_density: Tuple[float]
    measurements_count: int

    def __init__(self, *,
                 up_density: Iterable[float],
                 down_density: Iterable[float],
                 measurements_count: int) -> None:
        self.up_density = tuple(up_density)
        self.down_density = tuple(down_density)
        self.measurements_count = measurements_count

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls}

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass
class ExperimentRun:
    """Single Fermi-Hubbard circuit execution run data.

    Attributes:
        trotter_steps: Trotter steps count executed.
        end_timestamp_sec: Seconds since the unix epoch at the moment of circuit
            completion.
        result: Extracted fermionic number densities for each site.
        result_post_selected: Extracted fermionic number densities for each
            site with post selection applied.
        result_raw: Bitstring measurements for the circuit execution. This field
            is not populated if keep_raw_result argument of experiment_run
            function is set to False.
        calibration_timestamp_usec: Microseconds since the unix epoch that
            identify the processor calibration used during the circuit
            execution. This field is not populated by experiment_run function.
    """
    trotter_steps: int
    end_timestamp_sec: float
    result: ExperimentResult
    result_post_selected: ExperimentResult
    result_raw: Optional[cirq.Result] = None
    calibration_timestamp_usec: Optional[int] = None

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {
            cls.__name__: cls,
            **ExperimentResult.cirq_resolvers()
        }

    @property
    def calibration_timestamp_sec(self) -> float:
        return self.calibration_timestamp_usec / 1000.

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass
class FermiHubbardExperiment:
    """Results of a Fermi-Hubbard experiment execution for fixed parameters.

    Attributes:
        parameters: Problem parameters used for circuit compilation and
            execution.
        runs: Runs results for different Trotter step counts.
        processor: Processor name as passed to experiment_run function.
        name: Experiment name as passed to experiment_run function.
    """

    parameters: FermiHubbardParameters
    runs: Tuple[ExperimentRun]
    processor: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {
            cls.__name__: cls,
            **ExperimentRun.cirq_resolvers(),
            **FermiHubbardParameters.cirq_resolvers(),
        }

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


def save_experiment(experiment: FermiHubbardExperiment,
                    file_or_fn: Union[None, IO, pathlib.Path, str],
                    make_dirs: bool = True) -> None:
    """Persists experiment to a JSON file.

    Args:
        experiment: Experiment to save.
        file_or_fn: File description as passed to cirq.to_json function.
        make_dirs: When true and file_or_fn.
    """
    if isinstance(file_or_fn, str):
        dir = os.path.dirname(file_or_fn)
        if dir and make_dirs:
            os.makedirs(dir, exist_ok=True)

    cirq.to_json(experiment, file_or_fn)


def load_experiment(file_or_fn: Union[None, IO, pathlib.Path, str]
                    ) -> FermiHubbardExperiment:
    """Loads experiment from the JSON file.

    Args:
        file_or_fn: File description as passed to cirq.to_json function.
    """
    data = cirq.read_json(
        file_or_fn,
        resolvers=cirq.DEFAULT_RESOLVERS +
                  [FermiHubbardExperiment.cirq_resolvers().get])
    return cast(FermiHubbardExperiment, data)


def run_experiment(
        parameters: FermiHubbardParameters,
        steps: Iterable[int],
        sampler: cirq.Sampler = cirq.Simulator(),
        *,
        repetitions: int = 20000,
        name: Optional[str] = None,
        processor: Optional[str] = None,
        keep_raw_result: bool = False,
        threads: int = 4,
        pre_run_func: Optional[Callable[[int, cirq.Circuit], None]] = None,
        post_run_func: Optional[Callable[[int, cirq.Result], None]] = None
) -> FermiHubbardExperiment:
    """Executes Fermi-Hubbard experiment.

    Args:
        parameters: Fermi-Hubbard problem description.
        steps: Array of Trotter step counts that should be simulated.
        sampler: Sampler used to run the circuits. This can be either a 
            simulation sampler or sampler with capability of running on quantum
            hardware.
        repetitions: Number of repetitions for each circuit executed.
        name: Name which is passed to returned container.
        processor: Processor name which is passed to returned container.
        keep_raw_result: When true, the raw bitstring measurements for each run
            will be stored within the results object.
        threads: Number of threads used for execution. When set to 0, no
            multithreading routines are used.
        pre_run_func: Optional callable which is called before each circuit
            is scheduled for execution.
        post_run_func: Optional callable which is called after each circuit
            completes.

    Returns:
          Instance of FermiHubbardExperiment with experiment results.
    """

    def run_step(step: int) -> ExperimentRun:
        initial, trotter, measurement = create_circuits(parameters, step)
        circuit = initial + trotter + measurement

        if pre_run_func:
            pre_run_func(step, circuit)

        trial = sampler.run(circuit, repetitions=repetitions)
        end_time = time.time()

        if post_run_func:
            post_run_func(step, trial)

        result = extract_result(parameters.up_particles,
                                parameters.down_particles,
                                trial,
                                post_select=False)

        result_post_selected = extract_result(parameters.up_particles,
                                              parameters.down_particles,
                                              trial,
                                              post_select=True)

        return ExperimentRun(
            trotter_steps=step,
            end_timestamp_sec=end_time,
            result=result,
            result_post_selected=result_post_selected,
            result_raw=trial if keep_raw_result else None
        )

    if threads:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            runs = tuple(executor.map(run_step, steps))
    else:
        runs = tuple(run_step(step) for step in steps)

    return FermiHubbardExperiment(
        parameters=parameters,
        runs=runs,
        processor=processor,
        name=name
    )


def create_circuits(parameters: FermiHubbardParameters,
                    trotter_steps: int
                    ) -> Tuple[cirq.Circuit, cirq.Circuit, cirq.Circuit]:
    """

    Args:
        parameters: Fermi-Hubbard problem description.
        trotter_steps: Number of trotter steps to include.

    Returns:
        Tuple of:
        - initial state preparation circuit,
        - trotter steps circuit,
        - measurement circuit.
    """
    if isinstance(parameters.layout, LineLayout):
        return create_line_circuits(parameters, trotter_steps)
    elif isinstance(parameters.layout, ZigZagLayout):
        return create_zigzag_circuits(parameters, trotter_steps)
    else:
        raise ValueError(f'Unknown layout {parameters.layout}')


def extract_result(up_particles: int,
                   down_particles: int,
                   trial: cirq.Result,
                   post_select: bool) -> ExperimentResult:
    """Extracts accumulated results out of bitstring measurements.

    Args:
        up_particles: Number of particles in the up chain.
        down_particles: Number of particles in the down chain.
        trial: Bitstring measurements from quantum device or simulator.
        post_select: When true, post selection is used and only measurements
            which match the up_particles and down_particles counts in the up and
            down chain are used.
    """

    def post_selection(measurements: np.ndarray) -> np.ndarray:
        desired = np.array((up_particles, down_particles))
        counts = np.sum(measurements, axis=2, dtype=int)
        errors = np.abs(counts - desired)
        selected = measurements[(np.sum(errors, axis=1) == 0).nonzero()]
        if not len(selected):
            print(f'Warning: no measurement matching desired particle numbers'
                  f'{desired} for post selection {post_select}')
        return selected

    def create_result(measurements: np.ndarray) -> ExperimentResult:
        nd = np.average(measurements, axis=0)
        return ExperimentResult(
            measurements_count=len(measurements),
            up_density=nd[0, :],
            down_density=nd[1, :]
        )

    raw = trial.measurements['z']
    repetitions, qubits = raw.shape
    raw = raw.reshape((repetitions, 2, qubits // 2))

    if post_select:
        return create_result(post_selection(raw))
    else:
        return create_result(raw)
