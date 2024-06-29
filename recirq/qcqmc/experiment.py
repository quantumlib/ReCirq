# Copyright 2024 Google
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

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import attrs
import cirq
import numpy as np
import qsimcirq
from pytz import timezone

from recirq.qcqmc import blueprint, config, data, for_refactor


def _to_tuple(x: Optional[Iterable[float]]) -> Optional[Sequence[float]]:
    if x is None:
        return x
    return tuple(x)


@attrs.frozen(repr=False)
class SimulatedExperimentParams(data.Params):
    """Class for storing the parameters that specify an ExperimentData object.

    This stage of the experiment concerns itself with executing circuits and doing
    classical post-processing of shadow tomography data.

    Args:
        name: A `Params` name for this experiment.
        blueprint_params: Backreference to the `BlueprintParams` preceding this stage.
        n_samples_per_clifford: Number of circuit repetitions to take for each clifford.
        noise_model_name: For simulation; see `utilities.get_noise_model`.
        noise_model_params: For simulation; see `utilities.get_noise_model`.
        seed: The random seed to use for simulation.
        path_prefix: An optional path string prefix for the output files.
    """

    name: str
    blueprint_params: blueprint.BlueprintParams
    n_samples_per_clifford: int
    noise_model_name: str
    noise_model_params: Optional[Tuple[float, ...]] = attrs.field(
        converter=_to_tuple, default=None
    )
    seed: int = 0
    path_prefix: str = ""

    @property
    def path_string(self) -> str:
        return (
            self.path_prefix + config.OUTDIRS.DEFAULT_EXPERIMENT_DIRECTORY + self.name
        )

    def _json_dict_(self):
        simple_dict = attrs.asdict(self)
        simple_dict["blueprint_params"] = self.blueprint_params
        return simple_dict


@attrs.frozen(repr=False, eq=False)
class ExperimentData(data.Data):
    """The data defining the experimental result.

    Args:
        params: The experimental parameters.
        raw_samples: An array of shape [n_cliffords, n_samples_per_clifford, n_qubits]
            containing the bitstrings sampled from each of the different
            circuits.
        metadata: Any metadata associated with the run.
    """

    params: SimulatedExperimentParams
    raw_samples: np.ndarray = attrs.field(converter=np.asarray)
    metadata: Dict[str, Any] = attrs.field(factory=dict)

    def _json_dict_(self):
        simple_dict = attrs.asdict(self)
        simple_dict["params"] = self.params
        return simple_dict


def build_experiment(
    params: SimulatedExperimentParams, *, dependencies: Dict[data.Params, data.Data]
) -> ExperimentData:
    """Builds an ExperimentData from ExperimentParams

    Args:
        params: The experimental parameters.
        dependencies: The dependencies leading up to this point (in particular the blueprint.)
    """
    bp = dependencies[params.blueprint_params]
    assert isinstance(bp, blueprint.BlueprintData)
    assert params.blueprint_params == bp.params

    noise_model = for_refactor.get_noise_model(
        params.noise_model_name, params.noise_model_params
    )

    raw_samples = get_samples_from_simulation(
        bp.compiled_circuit,
        bp.resolvers,
        noise_model,
        params.n_samples_per_clifford,
        params.seed,
    )

    metadata = get_experimental_metadata()

    return ExperimentData(params=params, raw_samples=raw_samples, metadata=metadata)


def get_samples_from_simulation(
    circuit: cirq.Circuit,
    resolvers: List[cirq.ParamResolverOrSimilarType],
    noise_model: Union[None, cirq.NoiseModel],
    n_samples_per_clifford: int,
    seed: Optional[int] = None,
    simulate_single_precision: bool = config.SINGLE_PRECISION_DEFAULT,
) -> np.ndarray:
    """Samples the circuits and returns an array of sampled bits.

    Args:
        circuits: The shadow tomography circuits.
        resolvers: A list of cirq parameter resolvers.
        noise_model: An optional cirq.NoiseModel for the simulated experiment.
        n_samples_per_clifford: The number of samples to take per clifford sample.
        seed: An optional random seed
        simulate_single_precision: Use single precision instead of double
            precision for the circuit simulation.

    Returns:
        raw_samples: An array of shape [n_cliffords, n_samples_per_clifford,
            n_qubits] containing the bitstrings sampled from each of the different
            circuits.

    """
    simulator = qsimcirq.QSimSimulator()

    sampled_bitstrings = []

    if noise_model is not None:
        circuit = cirq.Circuit(
            noise_model.noisy_moments(circuit, sorted(circuit.all_qubits()))
        )

    for _, resolver in enumerate(resolvers):
        results = simulator.run(
            circuit, repetitions=n_samples_per_clifford, param_resolver=resolver
        )

        outcomes = results.measurements["all"]
        sampled_bitstrings.append(outcomes)

    raw_samples = np.stack(sampled_bitstrings)
    return raw_samples


def get_experimental_metadata() -> Dict[str, object]:
    """Gets some metadata to store along with the results for an experiment.

    Returns:
        A dictionary of useful metadata including the date and time of the experiment.
    """

    date_time = datetime.now()
    pacific_tz = timezone("US/Pacific")
    pacific_date_time = date_time.astimezone(pacific_tz)

    formatted_date_time = pacific_date_time.strftime("%m/%d/%Y, %H:%M:%S")

    metadata: Dict[str, Any] = {}
    metadata["PST_formatted_date_time"] = formatted_date_time
    metadata["iso_formatted_date_time"] = date_time.isoformat()

    return metadata
