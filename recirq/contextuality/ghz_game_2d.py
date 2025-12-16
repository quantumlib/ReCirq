# Copyright 2025 Google
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

r"""Functions and Classes for the 2D GHZ Game Experiment.

This module provides a complete pipeline for benchmarking the fidelity of
Greenberger-Horne-Zeilinger (GHZ) states on a 2D grid of qubits.
This is the code used for the implementation of the GHZ game detailed in arxiv.2512.02284

The GHZ Game:
The GHZ game is a non-communication game that can be won against classical
strategies if the players share an N-qubit GHZ state before playing the game.

The formal rules of the non-communication game are:
1. There are N players. Each player $j$ is given a classical input bit $x_j \\in \\{0, 1\\}$,
   such that the sum of all input bits is even ($\\sum_{j=1}^{N} x_j \\pmod 2 = 0$).
2. To win, the players must output bits $y_j \\in \\{0, 1\\}$ such that the parity of the
   output bits equals the parity of half the input sum:
   $$ \\sum_{j=1}^{N} y_j \\pmod 2 = \frac{1}{2} \\sum_{j=1}^{N} x_j \\pmod 2 $$

This module performs the following steps:

1. State Preparation: Creating the 2D GHZ state (using H and CZ gates).
2. Challenge Mapping: Assignments of the input bits $x_j$ (where $\\sum x_j$ is even) are
   mapped to local measurement settings: If $x_j=1$, measure in Y basis;
   if $x_j=0$, measure in X basis.
3. Run Experiment: Performing the measurement on the quantum computer with a sampler.
4. Winning Condition: The winning condition is met if the sum of the measured output
   bits ($\\sum y_j$) satisfies the parity condition derived from the challenge.

The experiment workflow is divided into two parts:

- **GHZ2dExperiment:** Creates and executes all necessary circuits
  (GHZ game trials and Z-stabilizer checks).
- **GHZ2dResults:** Stores the raw data and calculates the final fidelity metrics -
    win_mean, x- and z- type fidelity.
"""

import collections
import random
from typing import Any, cast

import cirq
import networkx as nx
import numpy as np
import warnings

_CIRCUIT_DD_SUPPORTED = hasattr(cirq, 'add_dynamical_decoupling')

if _CIRCUIT_DD_SUPPORTED:
    _ADD_DD_FUNC = cirq.add_dynamical_decoupling
else:
    _ADD_DD_FUNC = None

def _transform_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Transforms a Cirq circuit by applying a series of modifications.
    This is an internal helper function used exclusively by
    `generate_2d_ghz_circuit` when `add_dd_and_align_right` is True.

    The transformations for a circuit include:

    1. Adding a measurement to all qubits with a key 'm'.
       It serves as a stopping gate for the DD operation.
    2. Aligning the circuit and merging single-qubit gates.
    3. Stratifying the operations based on qubit count
    (1-qubit and 2-qubit gates).
    4. Applying dynamical decoupling to mitigate noise.
    5. Removing the final measurement operation to yield
       the state preparation circuit.

    Args:
        circuit: A cirq.Circuit object.

    Returns:
        The modified cirq.Circuit object.
    """
    qubits = list(circuit.all_qubits())
    circuit = circuit + cirq.Circuit(cirq.M(*qubits, key="m"))
    circuit = cirq.align_right(cirq.merge_single_qubit_gates_to_phxz(circuit))
    circuit = cirq.stratified_circuit(
        circuit[::-1], categories=[lambda op: cirq.num_qubits(op) == k for k in (1, 2)]
    )[::-1]
    if _CIRCUIT_DD_SUPPORTED:
        circuit = _ADD_DD_FUNC(circuit)
    else:
        warnings.warn(
            "Skipping Dynamical Decoupling: This feature requires Cirq 1.4 or later. "
            "Please upgrade your Cirq version to enable noise mitigation.",
            UserWarning
        )
    circuit = cirq.Circuit(circuit[:-1])
    return circuit


def generate_2d_ghz_circuit(
    center: cirq.GridQubit,
    graph: nx.Graph,
    num_qubits: int,
    randomized: bool = False,
    rng_or_seed: int | np.random.Generator | None = None,
    add_dd_and_align_right: bool = False,
) -> cirq.Circuit:
    """
    Generates a 2D GHZ state circuit with 'num_qubits'
    qubits using BFS method leveraged by NetworkX.
    The circuit is constructed by connecting qubits
    sequentially based on graph connectivity,
    starting from the 'center' qubit.
    The GHZ state is built using a series of H-CZ-H
    gate sequences.


    Args:
        center: The starting qubit for the GHZ state.
        graph: The connectivity graph of the qubits.
        num_qubits:  The number of qubits for the final
                     GHZ state. Must be greater than 0,
                     and less than or equal to
                     the total number of qubits
                     on the processor.
        randomized:  If True, neighbors are
                     added to the circuit in a random order.
                     If False, they are
                     are added by distance from the center.
        rng_or_seed: An optional seed or numpy random number
                     generator. Used only when randomized is True
        add_dd_and_align_right: If True, adds dynamical
                                decoupling and aligns right.

    Returns:
        A cirq.Circuit object for the GHZ state.

    Raises:
        ValueError: If num_qubits is non-positive or exceeds the total
                    number of qubits on the processor.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer.")

    if num_qubits > len(graph.nodes):
        raise ValueError(
            "num_qubits cannot exceed the total number of qubits on the processor."
        )

    if randomized:
        rng = (
            rng_or_seed
            if isinstance(rng_or_seed, np.random.Generator)
            else np.random.default_rng(rng_or_seed)
        )

        def sort_neighbors_fn(neighbors: list) -> list:
            """If 'randomized' is True, sort the neighbors randomly."""
            neighbors = list(neighbors)
            rng.shuffle(neighbors)
            return neighbors

    else:

        def sort_neighbors_fn(neighbors: list) -> list:
            """
            If 'randomized' is False, sort the neighbors as per
            distance from the center.
            """
            return sorted(
                neighbors,
                key=lambda q: (q.row - center.row) ** 2 + (q.col - center.col) ** 2,
            )

    bfs_tree = nx.bfs_tree(graph, center, sort_neighbors=sort_neighbors_fn)
    qubits_to_include = list(bfs_tree.nodes)[:num_qubits]
    final_tree = bfs_tree.subgraph(qubits_to_include)

    ops = []

    for node in nx.topological_sort(final_tree):
        # Handling the center qubit first
        if node == center:
            ops.append(cirq.H(node))
            continue

        for parent in final_tree.predecessors(node):
            ops.extend([cirq.H(node), cirq.CZ(parent, node), cirq.H(node)])

    circuit = cirq.Circuit(ops)

    if add_dd_and_align_right:
        return _transform_circuit(circuit)
    else:
        return circuit


class GHZ2dExperiment:
    """Generates and executes GHZ game circuits on a quantum sampler.

    This class combines the circuit generation and execution steps. It creates
    the necessary GHZ circuits, runs them on the provided sampler, and
    returns a GHZ2dResults object containing the analyzed data.
    """

    def __init__(self, graph: nx.Graph, center: cirq.GridQubit):
        """Initializes the experiment runner.

        Args:
            graph: The connectivity graph of the device.
            center: The starting qubit for the GHZ state generation.
        """
        self.graph = graph
        self.center = center

    @staticmethod
    def _get_circuit_to_append(a: np.ndarray, qubits: list[cirq.Qid]) -> cirq.Circuit:
        """Creates the measurement-setting circuit appended to the GHZ state preparation.

        This circuit applies Rx(pi/2) or Ry(-pi/2) to rotate the measurement basis
        from the Z-basis into the required X-Y plane basis for the GHZ game,
        followed by a joint measurement.

        Args:
            a: A numpy array of 0s and 1s, encoding the required rotation for each qubit.
            qubits: The list of qubits corresponding to the 'a' array indices.

        Returns:
            A cirq.Circuit object containing the final rotations and measurement.
        """
        return cirq.Circuit(
            cirq.Rx(rads=np.pi / 2).on_each(q for q, ai in zip(qubits, a) if ai == 1),
            cirq.Ry(rads=-np.pi / 2).on_each(q for q, ai in zip(qubits, a) if ai == 0),
            cirq.M(*qubits, key="m"),
        )

    def _generate_experiment_circuits(
        self,
        target_qubit_counts: list[int],
        num_trials_per_circuit: int,
        randomize_growth: bool = False,
        add_dd_and_align_right: bool = True,
        rng_or_seed: int | np.random.Generator | None = None,
    ) -> tuple[list[cirq.Circuit], list[dict[str, Any]]]:
        """Generates a master list of all individual circuits required for the batch run,
        including GHZ game trials (that are same for X-stabilizer) and Z-stabilizer
        circuits for each base circuit.

        This method generates circuits only for the specified qubit counts, at least 3, using the
        external `generate_2d_ghz_circuit`. Circuits requiring fewer than 3 qubits or more
        than the available device qubits are skipped. The logic ensures N is in the range
        [3, max_qubits_on_device] for meaningful GHZ game analysis.

        Args:
            target_qubit_counts: A list of integers specifying the number of qubits
                for which GHZ circuits should be generated (e.g., [3, 4, 6]).
                The minimum number of qubits should be 3.
            num_trials_per_circuit: The number of GHZ game trials to generate
                for each incremental base circuit (e.g., 20).
            randomize_growth: If True, the order of adding qubits to the GHZ state
                is randomized via the underlying generator function.
            add_dd_and_align_right: If True, applies transformations (DD, alignment)
                to the base GHZ circuit during generation.
            rng_or_seed: An optional seed or numpy random number generator, used
                if `randomize_growth` is True.

        Returns:
            A tuple of list of circuits and a list of metadata dictionaries for tracking purposes.
            This allows the experiment to handle arbitrary GHZ size in any order.
        """
        circuits: list[cirq.Circuit] = []
        metadata_list: list[dict[str, Any]] = []
        rng_game = np.random.default_rng()

        max_qubits_on_device = len(self.graph.nodes)

        for num_qubits in target_qubit_counts:
            if num_qubits <= 2 or num_qubits > max_qubits_on_device:
                print(
                    f"Warning: Skipping target qubit count {num_qubits}. Must be between 3 and "
                    f"{max_qubits_on_device}."
                )
                continue

            base_circuit = generate_2d_ghz_circuit(
                center=self.center,
                graph=self.graph,
                num_qubits=num_qubits,
                randomized=randomize_growth,
                rng_or_seed=rng_or_seed,
                add_dd_and_align_right=add_dd_and_align_right,
            )

            qubits = list(base_circuit.all_qubits())
            circuit_base_dropped_measurements = cirq.drop_terminal_measurements(
                base_circuit
            )

            a_all = rng_game.integers(0, 2, size=(num_trials_per_circuit, num_qubits))
            a_all[:, -1] = np.sum(a_all[:, :-1], axis=1) % 2
            assert np.all(np.sum(a_all, axis=1) % 2 == 0)

            for trial_idx in range(num_trials_per_circuit):
                a_values = a_all[trial_idx]
                appended_circuit = self._get_circuit_to_append(a_values, qubits)
                full_trial_circuit = (
                    circuit_base_dropped_measurements + appended_circuit
                )

                circuits.append(full_trial_circuit)
                metadata_list.append(
                    {
                        "type": "ghz_game",
                        "num_qubits": num_qubits,
                        "a_values": a_values,
                    }
                )

            circuits.append(
                circuit_base_dropped_measurements
                + cirq.Circuit(cirq.M(*qubits, key="m"))
            )
            metadata_list.append({"type": "z_stabilizer", "num_qubits": num_qubits})

        return circuits, metadata_list

    def run(
        self,
        target_qubit_counts: list[int],
        num_trials_per_circuit: int,
        sampler: cirq.Sampler,
        repetitions_for_batch: int = 1020,
        randomize_growth: bool = False,
        add_dd_and_align_right: bool = True,
        rng_or_seed: int | np.random.Generator | None = None,
    ) -> "GHZ2dResults":
        """Generates, runs, and analyzes the GHZ game experiment.
        This method generates circuits only for the specified qubit counts, at least 3.
        Circuits requiring fewer than 3 qubits or more
        than the available device qubits are skipped. The logic ensures N is in the range
        [3, max_qubits_on_device] for meaningful GHZ game analysis.

        Args:
            target_qubit_counts: A list of integers specifying the number of qubits
                for which GHZ circuits should be generated (e.g., [3, 4, 6]).
            num_trials_per_circuit: The number of GHZ game trials to generate
                for each base circuit (e.g., 20).
            sampler: The cirq.Sampler to run the circuits on.
            repetitions_for_batch: Number of measurement repetitions for each circuit run.
            randomize_growth: If True, the order of adding qubits to the GHZ state
                is randomized via the underlying generator function.
            add_dd_and_align_right: If True, applies transformations (DD, merge single qubit gates,
            and alignment) to the base GHZ circuit during generation.
            rng_or_seed: An optional seed or numpy random number generator, used
                if `randomize_growth` is True.

        Returns:
            A GHZ2dResults object containing the analyzed results.
        """
        circuits_original_order, metadata_list = self._generate_experiment_circuits(
            target_qubit_counts=target_qubit_counts,
            num_trials_per_circuit=num_trials_per_circuit,
            randomize_growth=randomize_growth,
            add_dd_and_align_right=add_dd_and_align_right,
            rng_or_seed=rng_or_seed,
        )

        original_indices = list(range(len(circuits_original_order)))
        shuffled_indices = original_indices.copy()
        random.shuffle(shuffled_indices)
        shuffled_circuits_to_run = [
            circuits_original_order[i] for i in shuffled_indices
        ]

        print(
            f"Running {len(shuffled_circuits_to_run)} circuits in a shuffled batch with "
            f"{repetitions_for_batch} repetitions each..."
        )

        results_batch_shuffled = sampler.run_batch(
            shuffled_circuits_to_run, repetitions=repetitions_for_batch
        )
        print("Batch run complete.")

        results_original_order = cast(list[cirq.Result], [None] * len(shuffled_indices))
        for i, shuffled_result in enumerate(results_batch_shuffled):
            original_idx = shuffled_indices[i]
            results_original_order[original_idx] = shuffled_result[0]

        return GHZ2dResults(
            metadata_list, results_original_order, num_trials_per_circuit
        )


class GHZ2dResults:
    """
    A unified container class that stores raw experimental data and calculates
    all fidelity and expectation value metrics upon initialization.

    Metrics are exposed as read-only properties for easy access, example: .win_mean

    Attributes:
        _metadata_list: List of dictionaries defining metadata for each circuit.
        _results_original_order: List of raw cirq.Result objects (unshuffled).
        _metrics: Dictionary holding all calculated
            metrics: {metric_name: {num_qubits: [list_of_values]}}.
    """

    def __init__(
        self,
        metadata_list: list[dict[str, Any]],
        results_original_order: list[cirq.Result],
        num_trials_per_circuit: int,
    ):
        """Initializes the results container and performs all analysis.

        Args:
            metadata_list: Metadata for all circuits executed.
            results_original_order: Raw results from the sampler in original order.
            num_trials_per_circuit: The number of GHZ game trials per base circuit.
        """
        self._metadata_list = metadata_list
        self._results_original_order = results_original_order
        self.num_trials_per_circuit = num_trials_per_circuit
        self._rng_analysis = np.random.default_rng()

        # Calculate metrics upon instantiation
        self._metrics = self._analyze()

    def _analyze(self) -> dict[str, dict[int, list]]:
        """
        Processes the raw measurement results and calculates all defined metrics.
        We noticed the first few shots were under-performing from the mean, so we
        discard first 20 shots.
        """
        results_metrics: dict[str, dict[int, list]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

        for i, res in enumerate(self._results_original_order):
            metadata = self._metadata_list[i]
            num_qubits = metadata["num_qubits"]
            circuit_type = metadata["type"]

            # Stabilization period requires dropping the first ~20 shots
            all_measurements_for_this_circuit = res.measurements["m"][20:]

            if circuit_type == "ghz_game":
                a_values = metadata["a_values"]
                expected_parity = (np.sum(a_values) // 2) % 2

                all_parities = np.sum(all_measurements_for_this_circuit, axis=1) % 2
                winning_repetitions = all_parities == expected_parity
                results_metrics["win_mean"][num_qubits].append(
                    np.mean(winning_repetitions)
                )

                transformed_measurements = 1 - 2 * all_measurements_for_this_circuit
                mean_product_raw = np.mean(np.prod(transformed_measurements, axis=1))

                if expected_parity == 1:
                    mean_product_raw = -1 * mean_product_raw
                results_metrics["x_fidelity"][num_qubits].append(mean_product_raw)

            elif circuit_type == "z_stabilizer":
                z_vals = 1 - 2 * all_measurements_for_this_circuit
                generators = [(i, i + 1) for i in range(num_qubits - 1)]
                alphas = self._rng_analysis.integers(
                    0, 2, size=(self.num_trials_per_circuit, num_qubits - 1)
                )

                for alpha in alphas:
                    stabilizer_qubit_indices: set[int] = set()
                    for bit, (i, j) in zip(alpha, generators):
                        if bit:
                            stabilizer_qubit_indices.symmetric_difference_update([i, j])

                    qubit_indices = sorted(stabilizer_qubit_indices)

                    if not qubit_indices:
                        mean_product_raw = 1.0
                    else:
                        selected_z_vals = z_vals[:, qubit_indices]
                        mean_product_raw = np.mean(np.prod(selected_z_vals, axis=1))
                    results_metrics["z_fidelity"][num_qubits].append(mean_product_raw)

        return dict(results_metrics)

    @property
    def win_mean(self) -> dict[int, list[float]]:
        """Average winning probability across all repetitions for each trial."""
        return self._metrics["win_mean"]

    @property
    def x_fidelity(self) -> dict[int, list[float]]:
        """Average X-basis stabilizer expectation value for each trial."""
        return self._metrics["x_fidelity"]

    @property
    def z_fidelity(self) -> dict[int, list[float]]:
        """Average Z-basis stabilizer expectation value for each trial."""
        return self._metrics["z_fidelity"]
