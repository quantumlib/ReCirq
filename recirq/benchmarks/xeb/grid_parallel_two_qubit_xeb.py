# Copyright 2020 The Cirq Developers
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
"""Parallel two-qubit cross-entropy benchmarking on a grid.

Cross-entropy benchmarking is a method of estimating the fidelity of quantum
gates by executing random quantum circuits containing those gates.
This experiment performs cross-entropy benchmarking of a two-qubit gate applied
to connected pairs of grid qubits. The qubit pairs are benchmarked in parallel
by executing circuits that act on many pairs simultaneously.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TYPE_CHECKING,
    Tuple,
    cast,
)
import collections
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import itertools
import multiprocessing
import os

import numpy as np

import cirq

from recirq.benchmarks.xeb.xeb_results import (
    CrossEntropyResult,
    CrossEntropyResultDict,
    CrossEntropyPair,
    SpecklePurityPair,
)

DEFAULT_BASE_DIR = os.path.expanduser(
    os.path.join("~", "cirq-results", "grid-parallel-two-qubit-xeb")
)

LAYER_A = cirq.experiments.GridInteractionLayer(
    col_offset=0, vertical=True, stagger=True
)
LAYER_B = cirq.experiments.GridInteractionLayer(
    col_offset=1, vertical=True, stagger=True
)
LAYER_C = cirq.experiments.GridInteractionLayer(
    col_offset=1, vertical=False, stagger=True
)
LAYER_D = cirq.experiments.GridInteractionLayer(
    col_offset=0, vertical=False, stagger=True
)

SINGLE_QUBIT_GATES = [
    cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(np.linspace(0, 7 / 4, 8), repeat=2)
]

GridQubitPair = Tuple["cirq.GridQubit", "cirq.GridQubit"]


def save(params: Any, obj: Any, base_dir: str, mode: str = "x") -> str:
    """Save an object to filesystem as a JSON file.

    Arguments:
        params: Parameters describing the object. This should have an `filename`
            attribute containing the filename with which to save the object.
        obj: The object to save.
        base_dir: The directory in which to save the object.
        mode: The mode with which to open the file to write. Defaults to 'x',
            which means that the save will fail if the file already exists.

    Returns:
        The full path to the saved JSON file.
    """
    filename = os.path.join(base_dir, params.filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode) as f:
        cirq.to_json(obj, f)
    return filename


def _grid_parallel_xeb_resolver(cirq_type: str) -> Type:
    if cirq_type == 'recirq.GridParallelXEBMetadata':
        return GridParallelXEBMetadata

def load(params: Any, base_dir: str) -> Any:
    """Load an object from a JSON file.

    Arguments:
        params: Parameters describing the object. This should have an `filename`
            attribute containing the filename of the saved object.
        base_dir: The directory from which to load the object.

    Returns: The loaded object.
    """
    filename = os.path.join(base_dir, params.filename)
    return cirq.read_json(filename,
                          resolvers=cirq.DEFAULT_RESOLVERS +
                          [_grid_parallel_xeb_resolver])


@dataclasses.dataclass
class GridParallelXEBMetadata:
    """Metadata for a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """

    qubits: Sequence["cirq.Qid"]
    two_qubit_gate: "cirq.Gate"
    num_circuits: int
    repetitions: int
    cycles: Sequence[int]
    layers: Sequence[cirq.experiments.GridInteractionLayer]
    seed: Optional[int]

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    @classmethod
    def _json_namespace_(cls):
        return 'recirq'

    def __repr__(self) -> str:
        return (
            "recirq.benchmarks.xeb.grid_parallel_two_qubit_xeb."
            "GridParallelXEBMetadata("
            f"qubits={self.qubits!r}, "
            f"two_qubit_gate={self.two_qubit_gate!r}, "
            f"num_circuits={self.num_circuits!r}, "
            f"repetitions={self.repetitions!r}, "
            f"cycles={self.cycles!r}, "
            f"layers={self.layers!r}, "
            f"seed={self.seed!r})"
        )


@dataclasses.dataclass
class GridParallelXEBMetadataParameters:
    """Parameters describing metadata for a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """

    data_collection_id: str

    @property
    def filename(self) -> str:
        return os.path.join(self.data_collection_id, "metadata.json")


@dataclasses.dataclass
class GridParallelXEBCircuitParameters:
    """Parameters describing a circuit used in a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
        layer: The grid layer specifying the pattern of two-qubit interactions
            in a layer of two-qubit gates.
        circuit_index: The index of the circuit.
    """

    data_collection_id: str
    layer: cirq.experiments.GridInteractionLayer
    circuit_index: int

    @property
    def filename(self) -> str:
        return os.path.join(
            self.data_collection_id,
            "circuits",
            f"{self.layer}",
            f"circuit-{self.circuit_index}.json",
        )


@dataclasses.dataclass
class GridParallelXEBTrialResultParameters:
    """Parameters describing a trial result from a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
        layer: The grid layer specifying the pattern of two-qubit interactions
            in a layer of two-qubit gates for the circuit used to obtain the
            trial result.
        depth: The number of cycles executed. A cycle consists of a layer of
            single-qubit gates followed by a layer of two-qubit gates.
        circuit_index: The index of the circuit from which the executed cycles
            are taken.
    """

    data_collection_id: str
    layer: cirq.experiments.GridInteractionLayer
    depth: int
    circuit_index: int

    @property
    def filename(self) -> str:
        return os.path.join(
            self.data_collection_id,
            "data",
            f"{self.layer}",
            f"circuit-{self.circuit_index}",
            f"depth-{self.depth}.json",
        )


@dataclasses.dataclass
class GridParallelXEBResultsParameters:
    """Parameters describing results from a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """

    data_collection_id: str

    @property
    def filename(self) -> str:
        return os.path.join(self.data_collection_id, "results.json")


def collect_grid_parallel_two_qubit_xeb_data(
    sampler: "cirq.Sampler",
    qubits: Iterable["cirq.GridQubit"],
    two_qubit_gate: "cirq.Gate",
    *,
    num_circuits: int = 50,
    repetitions: int = 10_000,
    cycles: Iterable[int] = range(3, 204, 10),
    layers: Sequence[cirq.experiments.GridInteractionLayer] = (
        LAYER_A,
        LAYER_B,
        LAYER_C,
        LAYER_D,
    ),
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    data_collection_id: Optional[str] = None,
    num_workers: int = 4,
    base_dir: str = DEFAULT_BASE_DIR,
) -> str:
    """Collect data for a grid parallel two-qubit XEB experiment.

    For each grid interaction layer in `layers`, `num_circuits` random circuits
    are generated. The circuits are generated using the function
    `cirq.experiments.random_rotations_between_grid_interaction_layers_circuit`,
    with two-qubit layer pattern consisting of only the corresponding grid
    interaction layer. Each random circuit is generated with a depth of
    `max(cycles)`. The single-qubit gates used are those of the form
    `cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)`
    with `z` and `a` each ranging through the set of 8 values
    [0, 1/4, ..., 7/4]. The final single-qubit gate layer is omitted.
    Note that since the same set of interactions is applied
    in each two-qubit gate layer, a circuit does not entangle pairs of qubits
    other than the pairs present in the corresponding grid interaction layer.

    Each circuit is used to generate additional circuits by truncating it at
    the cycle numbers specified by `cycles`. For instance, if `cycles` is
    [2, 4, 6], then the original circuit will consist of 6 cycles, and it will
    give rise to two additional circuits, one consisting of the first 2 cycles,
    and the other consisting of the first 4 cycles. The result is that the total
    number of generated circuits is `len(layers) * num_circuits * len(cycles)`.
    Each of these circuits is sampled with the number of repetitions specified
    by `repetitions`. The trial results of the circuit executions are saved to
    filesystem as JSON files. The full-length circuits are also saved to
    filesystem as JSON files. The resulting directory structure looks like

        {base_dir}
        └── {data_collection_id}
            ├── circuits
            │   ├── {layer0}
            │   │   ├── circuit-0.json
            │   │   ├── ...
            │   ├── ...
            ├── data
            │   ├── {layer0}
            │   │   ├── circuit-0
            │   │   │   ├── depth-{depth0}.json
            │   │   │   ├── ...
            │   │   ├── ...
            │   ├── ...
            └── metadata.json

    The `circuits` directory contains the circuits and the `data` directory
    contains the trial results. Both directories are split into subdirectories
    corresponding to the grid interaction layers. In `circuits`, these
    subdirectories contain the circuits as JSON files. In `data`, instead of
    having one file for each circuit, there is a directory for each circuit
    containing trial results for each cycle number, or depth.
    `metadata.json` saves the arguments passed to this function, other than
    `sampler`, `data_collection_id`, and `base_dir`. If `data_collection_id`
    is not specified, it is set to the current date and time.
    `base_dir` defaults to `~/cirq-results/grid-parallel-xeb`.

    Args:
        sampler: The quantum computer or simulator to use to run circuits.
        qubits: The qubits to use.
        two_qubit_gate: The two-qubit gate to use.
        num_circuits: The number of random circuits to generate.
        repetitions: The number of repetitions to sample.
        cycles: The cycle numbers at which to truncate the generated circuits.
        layers: The grid interaction layers to use.
        seed: A seed for the pseudorandom number generator, used to generate
            the random circuits.
        data_collection_id: The data collection ID to use. This determines the
            name of the directory in which data is saved to filesystem.
        num_workers: The maximum number of threads to use to run circuits
            concurrently.
        base_dir: The base directory in which to save data to filesystem.

    Side effects:
        Saves data to filesystem in the directory structure described above.
    """
    if data_collection_id is None:
        data_collection_id = datetime.datetime.now().isoformat().replace(":", "")
    qubits = list(qubits)
    cycles = list(cycles)
    prng = cirq.value.parse_random_state(seed)

    # Save metadata
    metadata_params = GridParallelXEBMetadataParameters(
        data_collection_id=data_collection_id
    )
    metadata = GridParallelXEBMetadata(  # type: ignore
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=num_circuits,
        repetitions=repetitions,
        cycles=cycles,
        layers=list(layers),
        seed=seed if isinstance(seed, int) else None,
    )
    save(metadata_params, metadata, base_dir=base_dir)

    # Generate and save all circuits
    max_cycles = max(cycles)
    circuits_: Dict[
        cirq.experiments.GridInteractionLayer, List[cirq.Circuit]
    ] = collections.defaultdict(list)
    for layer in layers:
        for i in range(num_circuits):
            circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
                qubits=qubits,
                depth=max_cycles,
                two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b),
                pattern=[layer],
                single_qubit_gates=SINGLE_QUBIT_GATES,
                add_final_single_qubit_layer=False,
                seed=prng,
            )
            circuits_[layer].append(circuit)
            circuit_params = GridParallelXEBCircuitParameters(
                data_collection_id=data_collection_id, layer=layer, circuit_index=i
            )
            save(circuit_params, circuit, base_dir=base_dir)

    # Collect data
    def run_truncated_circuit(
        truncated_circuit: "cirq.Circuit",
        layer: cirq.experiments.GridInteractionLayer,
        depth: int,
        circuit_index: int,
    ) -> None:
        print(f"\tSampling from circuit {circuit_index} of layer {layer}.")
        truncated_circuit.append(cirq.measure(*qubits, key="m"))
        trial_result = sampler.run(truncated_circuit, repetitions=repetitions)
        trial_result_params = GridParallelXEBTrialResultParameters(
            data_collection_id=cast(str, data_collection_id),
            layer=layer,
            depth=depth,
            circuit_index=circuit_index,
        )
        save(trial_result_params, trial_result, base_dir=base_dir)

    for depth in cycles:
        print(f"Executing circuits at depth {depth}.")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for layer in layers:
                truncated_circuits = [
                    circuit[: 2 * depth] for circuit in circuits_[layer]
                ]
                for i, truncated_circuit in enumerate(truncated_circuits):
                    executor.submit(
                        run_truncated_circuit, truncated_circuit, layer, depth, i
                    )

    return data_collection_id


def compute_grid_parallel_two_qubit_xeb_results(
    data_collection_id: str, num_processors: int = 1, base_dir: str = DEFAULT_BASE_DIR
) -> CrossEntropyResultDict:
    """Compute grid parallel two-qubit XEB results from experimental data.

    XEB fidelities are calculated using the least squares XEB fidelity estimator
    with the linear cross entropy observable. Using notation from the docstring
    of `_least_squares_xeb_fidelity_from_expectations`,
    the linear cross entropy observable O_U has eigenvalue corresponding to the
    computational basis state |z⟩ given by D * |⟨z|𝜓_U⟩|^2.

    Purity values are calculated using speckle purity benchmarking. For details,
    see the docstring of the method
    `cirq.experiments.purity_from_probabilities`.

    Results are saved in the file {base_dir}/{data_collection_id}/results.json.

    Args:
        data_collection_id: The data collection ID of the data set to analyze.
        num_processors: The number of CPUs to use.
        base_dir: The base directory from which to read data.

    Returns:
        CrossEntropyResultDict mapping qubit pairs to XEB results.

    Side effects:
        Saves the returned CrossEntropyResultDict to the file
        {base_dir}/{data_collection_id}/results.json.
    """
    metadata_params = GridParallelXEBMetadataParameters(
        data_collection_id=data_collection_id
    )
    metadata = load(metadata_params, base_dir=base_dir)
    qubits = metadata.qubits
    num_circuits = metadata.num_circuits
    repetitions = metadata.repetitions
    cycles = metadata.cycles
    layers = metadata.layers

    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)
    all_active_qubit_pairs: List[GridQubitPair] = []
    qubit_indices = {q: i for i, q in enumerate(qubits)}
    xeb_results: Dict[GridQubitPair, CrossEntropyResult] = {}

    # Load data into a dictionary mapping qubit pair to list of
    # (circuit, measurement_results) tuples
    data: Dict[
        GridQubitPair, List[Tuple[cirq.Circuit, List[np.ndarray]]]
    ] = collections.defaultdict(list)
    for layer in layers:
        active_qubit_pairs = [pair for pair in coupled_qubit_pairs if pair in layer]
        all_active_qubit_pairs.extend(active_qubit_pairs)
        for i in range(num_circuits):
            circuit_params = GridParallelXEBCircuitParameters(
                data_collection_id=data_collection_id, layer=layer, circuit_index=i
            )
            circuit = load(circuit_params, base_dir=base_dir)
            trial_results = []
            for depth in cycles:
                trial_result_params = GridParallelXEBTrialResultParameters(
                    data_collection_id=data_collection_id,
                    layer=layer,
                    depth=depth,
                    circuit_index=i,
                )
                trial_result = load(trial_result_params, base_dir=base_dir)
                trial_results.append(trial_result)
            for qubit_pair in active_qubit_pairs:
                # Restrict measurements to this qubit pair
                a, b = qubit_pair
                indices = [qubit_indices[a], qubit_indices[b]]
                restricted_measurement_results = []
                for trial_result in trial_results:
                    # Get the measurements of this qubit pair
                    restricted_measurements = trial_result.measurements["m"][:, indices]
                    # Convert length-2 bitstrings to integers
                    restricted_measurements = (
                        2 * restricted_measurements[:, 0]
                        + restricted_measurements[:, 1]
                    )
                    restricted_measurement_results.append(restricted_measurements)
                data[qubit_pair].append(
                    (circuit[:, qubit_pair], restricted_measurement_results)
                )

    # Compute the XEB results
    arguments = []
    for qubit_pair in all_active_qubit_pairs:
        circuits, measurement_results = zip(*data[qubit_pair])
        arguments.append(
            (
                qubit_pair,
                circuits,
                measurement_results,
                num_circuits,
                repetitions,
                cycles,
            )
        )
    num_processors = min(num_processors, len(arguments))
    with multiprocessing.Pool(num_processors) as pool:
        xeb_result_list = pool.starmap(_get_xeb_result, arguments)
    xeb_results = {
        qubit_pair: result
        for qubit_pair, result in zip(all_active_qubit_pairs, xeb_result_list)
    }

    # Save results and return
    result_dict = CrossEntropyResultDict(results=xeb_results)  # type: ignore
    result_params = GridParallelXEBResultsParameters(
        data_collection_id=data_collection_id
    )
    save(result_params, result_dict, base_dir=base_dir)

    return result_dict


def _get_xeb_result(
    qubit_pair: GridQubitPair,
    circuits: List["cirq.Circuit"],
    measurement_results: Sequence[List[np.ndarray]],
    num_circuits: int,
    repetitions: int,
    cycles: List[int],
) -> CrossEntropyResult:
    # pytest-cov is unable to detect that this function is called by a
    # multiprocessing Pool
    # coverage: ignore
    simulator = cirq.Simulator()
    # Simulate circuits to get bitstring probabilities
    all_and_observed_probabilities: Dict[
        int, List[Tuple[np.ndarray, np.ndarray]]
    ] = collections.defaultdict(list)
    empirical_probabilities: Dict[int, List[np.ndarray]] = collections.defaultdict(list)
    for i, circuit in enumerate(circuits):
        step_results = simulator.simulate_moment_steps(circuit, qubit_order=qubit_pair)
        moment_index = 0
        for depth, measurements in zip(cycles, measurement_results[i]):
            while moment_index < 2 * depth:
                step_result = next(step_results)
                moment_index += 1
            amplitudes = step_result.state_vector()
            probabilities = cirq.state_vector_to_probabilities(amplitudes)
            _, counts = np.unique(measurements, return_counts=True)
            empirical_probs = counts / len(measurements)
            empirical_probs = np.pad(
                empirical_probs, (0, 4 - len(empirical_probs)), mode="constant"
            )
            all_and_observed_probabilities[depth].append(
                (probabilities, probabilities[measurements])
            )
            empirical_probabilities[depth].append(empirical_probs)
    # Compute XEB result
    data = []
    purity_data = []
    for depth in cycles:
        all_probabilities, observed_probabilities = zip(
            *all_and_observed_probabilities[depth]
        )
        empirical_probs = np.asarray(empirical_probabilities[depth]).flatten()
        # TODO: replace this with cirq.experiments.xeb_fitting if possible.
        fidelity, _ = _least_squares_xeb_fidelity_from_probabilities(
            hilbert_space_dimension=4,
            observed_probabilities=observed_probabilities,
            all_probabilities=all_probabilities,
            observable_from_probability=None,
            normalize_probabilities=True,
        )
        purity = cirq.experiments.purity_from_probabilities(4, empirical_probs)
        data.append(CrossEntropyPair(depth, fidelity))
        purity_data.append(SpecklePurityPair(depth, purity))
    return CrossEntropyResult(  # type: ignore
        data=data, repetitions=repetitions, purity_data=purity_data
    )


def _coupled_qubit_pairs(qubits: List["cirq.GridQubit"]) -> List[GridQubitPair]:
    """Get pairs of GridQubits that are neighbors."""
    pairs = []
    qubit_set = set(qubits)
    for qubit in qubits:

        def add_pair(neighbor: "cirq.GridQubit"):
            if neighbor in qubit_set:
                pairs.append((qubit, neighbor))

        add_pair(cirq.GridQubit(qubit.row, qubit.col + 1))
        add_pair(cirq.GridQubit(qubit.row + 1, qubit.col))

    return pairs


def _least_squares_xeb_fidelity_from_expectations(
    measured_expectations: Sequence[float],
    exact_expectations: Sequence[float],
    uniform_expectations: Sequence[float],
) -> Tuple[float, List[float]]:
    """Least squares fidelity estimator.
    An XEB experiment collects data from the execution of random circuits
    subject to noise. The effect of applying a random circuit with unitary U is
    modeled as U followed by a depolarizing channel. The result is that the
    initial state |𝜓⟩ is mapped to a density matrix ρ_U as follows:
        |𝜓⟩ → ρ_U = f |𝜓_U⟩⟨𝜓_U| + (1 - f) I / D
    where |𝜓_U⟩ = U|𝜓⟩, D is the dimension of the Hilbert space, I / D is the
    maximally mixed state, and f is the fidelity with which the circuit is
    applied. Let O_U be an observable that is diagonal in the computational
    basis. Then the expectation of O_U on ρ_U is given by
        Tr(ρ_U O_U) = f ⟨𝜓_U|O_U|𝜓_U⟩ + (1 - f) Tr(O_U / D).
    This equation shows how f can be estimated, since Tr(ρ_U O_U) can be
    estimated from experimental data, and ⟨𝜓_U|O_U|𝜓_U⟩ and Tr(O_U / D) can be
    computed numerically.
    Let e_U = ⟨𝜓_U|O_U|𝜓_U⟩, u_U = Tr(O_U / D), and m_U denote the experimental
    estimate of Tr(ρ_U O_U). Then we estimate f by performing least squares
    minimization of the quantity
        f (e_U - u_U) - (m_U - u_U)
    over different random circuits (giving different U). The solution to the
    least squares problem is given by
        f = (∑_U (m_U - u_U) * (e_U - u_U)) / (∑_U (e_U - u_U)^2).
    Args:
        measured_expectations: A sequence of the m_U, the experimental estimates
            of the observable, one for each circuit U.
        exact_expectations: A sequence of the e_U, the exact value of the
            observable. The order should match the order of the
            `measured_expectations` argument.
        uniform_expectations: A sequence of the u_U, the expectation of the
            observable on a uniformly random bitstring. The order should match
            the order in the other arguments.
    Returns:
        A tuple of two values. The first value is the estimated fidelity.
        The second value is a list of the residuals
            f (e_U - u_U) - (m_U - u_U)
        of the least squares minimization.
    Raises:
        ValueError: The lengths of the input sequences are not all the same.
    """
    if not (len(measured_expectations) == len(exact_expectations) == len(uniform_expectations)):
        raise ValueError(
            'The lengths of measured_expectations, '
            'exact_expectations, and uniform_expectations must '
            'all be the same. Got lengths '
            f'{len(measured_expectations)}, '
            f'{len(exact_expectations)}, and '
            f'{len(uniform_expectations)}.'
        )
    numerator = 0.0
    denominator = 0.0
    for m, e, u in zip(measured_expectations, exact_expectations, uniform_expectations):
        numerator += (m - u) * (e - u)
        denominator += (e - u) ** 2
    fidelity = numerator / denominator
    residuals = [
        fidelity * (e - u) - (m - u)
        for m, e, u in zip(measured_expectations, exact_expectations, uniform_expectations)
    ]
    return fidelity, residuals


def _least_squares_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    observed_probabilities: Sequence[Sequence[float]],
    all_probabilities: Sequence[Sequence[float]],
    observable_from_probability: Optional[Callable[[float], float]] = None,
    normalize_probabilities: bool = True,
) -> Tuple[float, List[float]]:
    """Least squares fidelity estimator with observable based on probabilities.
    Using the notation from the docstring of
    `least_squares_xeb_fidelity_from_expectations`, this function computes the
    least squares fidelity estimate when the observable O_U has eigenvalue
    corresponding to the computational basis state |z⟩ given by g(p(z)), where
    p(z) = |⟨z|𝜓_U⟩|^2 and g is a function that can be specified. By default,
    g is the identity function, but other choices, such as the logarithm, are
    useful. By default, the probability p(z) is actually multiplied by the
    Hilbert space dimension D, so that the observable is actually g(D * p(z)).
    This behavior can be disabled by setting `normalize_probabilities` to
    False.
    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        observed_probabilities: Ideal probabilities of bitstrings observed in
            experiments. A list of lists, where each inner list contains the
            probabilities for a single circuit.
        all_probabilities: Ideal probabilities of all possible bitstrings.
            A list of lists, where each inner list contains the probabilities
            for a single circuit, and should have length equal to the Hilbert
            space dimension. The order of the lists should correspond to that
            of `observed_probabilities`.
        observable_from_probability: Function that computes the observable from
            a given probability.
        normalize_probabilities: Whether to multiply the probabilities by the
            Hilbert space dimension before computing the observable.
    Returns:
        A tuple of two values. The first value is the estimated fidelity.
        The second value is a list of the residuals
            f (e_U - u_U) - (m_U - u_U)
        of the least squares minimization.
    """
    if not isinstance(observable_from_probability, np.ufunc):
        if observable_from_probability is None:
            observable_from_probability = lambda p: p
        else:
            observable_from_probability = np.frompyfunc(observable_from_probability, 1, 1)
    observable_from_probability = cast(Callable, observable_from_probability)
    measured_expectations = []
    exact_expectations = []
    uniform_expectations = []
    prefactor = hilbert_space_dimension if normalize_probabilities else 1.0
    for observed_probs, all_probs in zip(observed_probabilities, all_probabilities):
        observable = observable_from_probability(prefactor * np.array(all_probs))
        measured_expectations.append(
            np.mean(observable_from_probability(prefactor * np.array(observed_probs)))
        )
        exact_expectations.append(np.sum(all_probs * observable))
        uniform_expectations.append(np.sum(observable) / hilbert_space_dimension)
    return _least_squares_xeb_fidelity_from_expectations(
        measured_expectations, exact_expectations, uniform_expectations
    )
