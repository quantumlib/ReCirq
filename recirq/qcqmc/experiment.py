from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cirq
import numpy as np
import qsimcirq
from pytz import timezone

from recirq.qcqmc.blueprint import BlueprintData, BlueprintParams
from recirq.qcqmc.utilities import (
    Data,
    get_noise_model,
    OUTDIRS,
    Params,
    SINGLE_PRECISION_DEFAULT,
)


@dataclass(frozen=True, repr=False)
class SimulatedExperimentParams(Params):
    """Class for storing the parameters that specify a ExperimentData.

    This stage of the experiment concerns itself with executing circuits and doing
    classical post-processing of shadow tomography data.


    Args:
        name: A `Params` name for this experiment.
        blueprint_params: Backreference to the `BlueprintParams` preceding this stage.
        n_samples_per_clifford: Number of circuit repetitions to take for each clifford.
        noise_model_name: For simulation; see `utilities.get_noise_model`.
        noise_model_params: For simulation; see `utilities.get_noise_model`.
        seed: The random seed to use for simulation.
    """

    name: str
    blueprint_params: BlueprintParams
    n_samples_per_clifford: int
    noise_model_name: str
    noise_model_params: Optional[Tuple[float, ...]] = None
    seed: int = 0

    def __post_init__(self):
        """A little special sauce to make sure that these end up as a tuple."""
        if self.noise_model_params is not None:
            object.__setattr__(
                self, "noise_model_params", tuple(self.noise_model_params)
            )

    @property
    def path_string(self) -> str:
        return OUTDIRS.DEFAULT_EXPERIMENT_DIRECTORY + self.name

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(frozen=True, eq=False)
class ExperimentData(Data):
    params: SimulatedExperimentParams
    raw_samples: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """We need to make raw_samples into an np.ndarray if it isn't provided that way."""
        object.__setattr__(self, "raw_samples", np.asarray(self.raw_samples))

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


def build_experiment(
    params: SimulatedExperimentParams, *, dependencies: Dict[Params, Data]
) -> ExperimentData:
    """Builds an ExperimentData from ExperimentParams"""
    blueprint = dependencies[params.blueprint_params]
    assert isinstance(blueprint, BlueprintData)
    assert params.blueprint_params == blueprint.params

    noise_model = get_noise_model(params.noise_model_name, params.noise_model_params)

    raw_samples = get_samples_from_simulation(
        blueprint.compiled_circuit,
        blueprint.resolvers,
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
    seed: Optional[int],
    simulate_single_precision: bool = SINGLE_PRECISION_DEFAULT,
) -> np.ndarray:
    """Samples the circuits and returns an array of sampled bits.

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

    for i, resolver in enumerate(resolvers):
        results = simulator.run(
            circuit, repetitions=n_samples_per_clifford, param_resolver=resolver
        )

        outcomes = results.measurements["all"]
        sampled_bitstrings.append(outcomes)

    raw_samples = np.stack(sampled_bitstrings)
    return raw_samples


def get_experimental_metadata() -> Dict[str, object]:
    """Gets some metadata to store along with the results for an experiment."""

    date_time = datetime.now()
    pacific_tz = timezone("US/Pacific")
    pacific_date_time = date_time.astimezone(pacific_tz)

    formatted_date_time = pacific_date_time.strftime("%m/%d/%Y, %H:%M:%S")

    metadata: Dict[str, Any] = {}
    metadata["PST_formatted_date_time"] = formatted_date_time
    metadata["iso_formatted_date_time"] = date_time.isoformat()

    return metadata
