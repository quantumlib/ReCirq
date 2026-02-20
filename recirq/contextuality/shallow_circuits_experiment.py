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

"""Builds and runs experiment for finding hidden linear functions (HLF) with shallow circuits.

This module provides the fundamental building blocks for generating, executing,
and analyzing results for finding HLF with shallow circuits experiment,
as described in https://arxiv.org/abs/2512.02284.
"""

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import cirq
import networkx as nx
import numpy as np
import stim
import stimcirq
from tqdm import tqdm


@dataclass
class HLFCircuit:
    """A container for a randomly generated HLF circuit and its metadata.

    Attributes:
        circuit: The generated `cirq.Circuit` object.
        total_dof: total degree of freedom (dof = num_qubits + total possible CZ gates)
            for classical layer analysis.
        original_cz_layers: A list of lists, representing the CZ gates in their
            original four layers before any dropout was applied.
        final_cz_layers: A list of lists, representing the CZ gates that
            remained after dropout, structured into their final layers.
        actual_cz_dropout_fraction: The actual fraction (0.0 to 1.0) of CZ
            gates that were dropped from the original set.
        actual_s_dropout_fraction: The actual fraction (0.0 to 1.0) of S
            gates that were dropped.
    """

    circuit: cirq.Circuit
    total_dof: int
    original_cz_layers: list[list[cirq.Operation]]
    final_cz_layers: list[list[cirq.Operation]]
    actual_cz_dropout_fraction: float
    actual_s_dropout_fraction: float


def _get_qubits_2d_on_device(
    n: int, all_qubits: list[cirq.GridQubit], all_edges: list[tuple[cirq.GridQubit, cirq.GridQubit]]
) -> list[cirq.GridQubit]:
    """Selects a compact, connected patch of n qubits using a BFS-based approach.

    Args:
        n: The number of qubits to select.
        all_qubits: A list of all available `cirq.GridQubit`s on the device.
        all_edges: A list of all coupled pairs on the device.

    Returns:
        A list of the selected `cirq.GridQubit`s.

    Raises:
        ValueError: If the requested number of qubits `n` is greater than the
            total number of available qubits on the device.
    """
    if n > len(all_qubits):
        raise ValueError(f"Requested {n} qubits, but device only has {len(all_qubits)}.")

    graph = nx.Graph(all_edges)

    # Find the geometric center of the device
    center_coord = np.mean([[q.row, q.col] for q in all_qubits], axis=0)

    # Find the qubit closest to the geometric center to use as the starting seed
    start_node = min(
        all_qubits, key=lambda q: (q.row - center_coord[0]) ** 2 + (q.col - center_coord[1]) ** 2
    )

    # Grow a patch of n qubits using a Breadth-First Search from the start node
    patch = list(nx.bfs_tree(graph, source=start_node))

    return patch[:n]


def hlf_circuit_biased_dropout(
    n_qubits: int,
    all_qubits: list[cirq.GridQubit],
    all_edges: list[tuple[cirq.GridQubit, cirq.GridQubit]],
    cz_dropout_fraction: float = 0.5,
    s_dropout_fraction: float = 0.5,
    seed: int | None = None,
) -> HLFCircuit:
    """Generates a shallow Clifford circuit on a grid with a random dropout fraction for S and CZ gates.

    Args:
        n_qubits: The number of qubits for the circuit.
        all_qubits: A list of all available `cirq.GridQubit`s on the device.
        all_edges: A list of all coupled pairs on the device.
        cz_dropout_fraction: The probability of dropping a CZ gate.
        s_dropout_fraction: The probability of dropping an S gate.
        seed: A random seed for reproducibility.

    Returns:
        An HLFCircuit object containing the generated circuit and metadata.
    """

    rng = random.Random(seed)

    qubits = _get_qubits_2d_on_device(n_qubits, all_qubits, all_edges)
    qubit_set = set(qubits)
    circuit = cirq.Circuit()

    # Moment 1: Initial Hadamards
    circuit.append(cirq.Moment(cirq.H.on_each(qubits)))

    # Moment 2: S gates with global dropout
    num_s_to_drop = int(n_qubits * s_dropout_fraction)
    s_indices_to_drop = set(rng.sample(range(n_qubits), num_s_to_drop))

    s_ops = []
    s_bits = []
    for i, q in enumerate(qubits):
        if i not in s_indices_to_drop:
            s_ops.append(cirq.S(q))
            s_bits.append(1)
        else:
            s_bits.append(0)

    if s_ops:
        circuit.append(cirq.Moment(s_ops))
    actual_s_dropout_fraction = (1 - sum(s_bits) / n_qubits) if n_qubits > 0 else 0

    # Extract valid CZ edges from topology
    valid_edges = [(q1, q2) for q1, q2 in all_edges if q1 in qubit_set and q2 in qubit_set]

    # Build 4 disjoint CZ layers
    original_layers: list[list[cirq.Operation]] = [[] for _ in range(4)]
    for q1, q2 in valid_edges:
        if q1.row == q2.row:
            parity = min(q1.col, q2.col) % 2
            original_layers[parity].append(cirq.CZ(q1, q2))  # Horizontal
        elif q1.col == q2.col:
            parity = min(q1.row, q2.row) % 2
            original_layers[2 + parity].append(cirq.CZ(q1, q2))  # Vertical

    # Global CZ dropout
    all_cz = sum(original_layers, [])
    total_cz = len(all_cz)
    num_to_drop = int(total_cz * cz_dropout_fraction)
    indices_to_drop = set(rng.sample(range(total_cz), num_to_drop))
    dropped_bits_global = [0 if i in indices_to_drop else 1 for i in range(total_cz)]

    final_layers = []
    counter = 0
    for layer in original_layers:
        final_layer = []
        for gate in layer:
            if dropped_bits_global[counter]:
                final_layer.append(gate)
            counter += 1
        final_layers.append(final_layer)

    actual_cz_dropout_fraction = num_to_drop / total_cz if total_cz > 0 else 0

    # Add CZ layers
    for layer in final_layers:
        if layer:
            circuit.append(cirq.Moment(layer))

    # Final Hadamards
    circuit.append(cirq.Moment(cirq.H.on_each(qubits)))

    # Measurement
    circuit.append(cirq.measure(*sorted(qubits), key="m"))

    # Calculate the total degree of freedom (dof) for classical layer analysis
    total_dof = total_cz + n_qubits

    return HLFCircuit(
        circuit=circuit,
        total_dof=total_dof,
        original_cz_layers=original_layers,
        final_cz_layers=final_layers,
        actual_cz_dropout_fraction=actual_cz_dropout_fraction,
        actual_s_dropout_fraction=actual_s_dropout_fraction,
    )


class ShallowCircuitExperiment:
    """Generates circuits and runs the HLF shallow circuits experiment.

    This class holds the configuration for a shallow circuit experiment,
    including the device topology and parameters for circuit generation.

    Args:
        all_qubits: A list of all available `cirq.GridQubit`s on the device.
        all_edges: A list of all coupled pairs of qubits on the device.
        n_qubits_list: A list of qubit counts for which to generate circuits.
        s_dropout_fraction: The dropout rate for S gates during generation.
        cz_dropout_fraction: The dropout rate for CZ gates during generation.
        n_runs: The number of random circuits to generate for each qubit count.

     Attributes:
        circuits: The list of `cirq.Circuit` objects for the experiment.
        n_qubits_map: A map of circuit index to qubit count.
        total_dof: A dict of total degree of freedom (dof = num_qubits + total possible CZ gates)
            for each n_qubits for classical layer analysis.
    """

    def __init__(
        self,
        all_qubits: list[cirq.GridQubit],
        all_edges: list[tuple[cirq.GridQubit, cirq.GridQubit]],
        n_qubits_list: list[int],
        s_dropout_fraction: float,
        cz_dropout_fraction: float,
        n_runs: int,
    ):
        self.all_qubits = all_qubits
        self.all_edges = all_edges
        self.n_qubits_list = n_qubits_list
        self.s_dropout_fraction = s_dropout_fraction
        self.cz_dropout_fraction = cz_dropout_fraction
        self.n_runs = n_runs

        # Attributes to hold the generated circuits, their map, and dof
        self.circuits: list[cirq.Circuit] | None = None
        self.n_qubits_map: list[int] | None = None
        self.total_dof: dict[int, int] | None = None

    def generate_circuits_for_experiment(self) -> None:
        """Generates circuits, n_qubits_map, and total_dof for a range of qubit numbers."""
        all_circuits = []
        n_qubits_map = []
        total_dof: dict[int, int] = {}

        print("Generating circuits...")
        for n_qubits in tqdm(self.n_qubits_list, desc="Qubit Counts"):
            for _ in range(self.n_runs):
                result_circuit = hlf_circuit_biased_dropout(
                    n_qubits=n_qubits,
                    all_qubits=self.all_qubits,
                    all_edges=self.all_edges,
                    cz_dropout_fraction=self.cz_dropout_fraction,
                    s_dropout_fraction=self.s_dropout_fraction,
                    seed=int.from_bytes(os.urandom(4), "big"),
                )
                all_circuits.append(result_circuit.circuit)
                n_qubits_map.append(n_qubits)

            # DOF is fixed for a given number of qubits
            total_dof[n_qubits] = result_circuit.total_dof

        self.circuits = all_circuits
        self.n_qubits_map = n_qubits_map
        self.total_dof = total_dof

        print(f"Generated {len(all_circuits)} circuits in total.")

    def run(self, sampler: cirq.Sampler, n_repetitions: int) -> "ShallowCircuitResults":
        """Runs the full experiment: generates circuits, runs them, and returns a results object.

        Args:
            sampler: The `cirq.Sampler` to use for execution.
            n_repetitions: The number of measurement repetitions for each circuit.

        Returns:
            A `ShallowCircuitsResults` object containing the results.
        """

        if self.circuits is None:
            self.generate_circuits_for_experiment()

        assert self.circuits is not None
        assert self.n_qubits_map is not None
        assert self.total_dof is not None

        print(f"Running {len(self.circuits)} circuits...")
        results = sampler.run_batch(self.circuits, repetitions=n_repetitions)

        unpacked_measurements = [result[0].measurements["m"] for result in results]

        print("Circuit execution complete.")

        return ShallowCircuitResults(
            circuits=self.circuits,
            measurements=unpacked_measurements,
            n_qubits_map=self.n_qubits_map,
            total_dof=self.total_dof,
        )


class ShallowCircuitResults:
    """A class that analyzes and stores the results.

    This class holds all the data generated by a `ShallowCircuitExperiment` run,
    including the circuits that were executed, the measurement outcomes, and
    associated metadata. It is responsible for analyzing these results to
    compute the fidelity of the bitstring.


    Args:
        circuits: The list of `cirq.Circuit` objects that were executed.
        measurements: A list of numpy arrays, where each array contains the
            measurement outcomes for the corresponding circuit.
        n_qubits_map: A list that maps each circuit in the `circuits` list
            to its number of qubits.
        total_dof: A dict of total degree of freedom (dof = num_qubits + total possible CZ gates)
            for each n_qubits for classical layer analysis

    Attributes:
        _probability_of_valid_outcomes: Dict mapping `n_qubits` to a list of fidelity scores
            after matching the outcome of the experiment to the ideal outcome.
        _effective_layers: Dict mapping `n_qubits` to a list of effective layer
            counts. Calculated as `4 / probability`, this estimates the
            equivalent number of noisy layers the hardware executed.
        _classical_layers: Dict mapping `n_qubits` to the classical degrees of
            freedom (DOF) for each circuit. Represents the problem complexity,
            calculated as log2(total_dof = num_qubits + total possible CZ gates).
    """

    def __init__(
        self,
        circuits: list[cirq.Circuit],
        measurements: list[np.ndarray],
        n_qubits_map: list[int],
        total_dof: dict[int, int],
    ):
        self.circuits = circuits
        self.measurements = measurements
        self.n_qubits_map = n_qubits_map
        self.total_dof = total_dof

        # Attribute
        self._probability_of_valid_outcomes: dict[int, list[float]] | None = None
        self._effective_layers: dict[int, list[float]] | None = None
        self._classical_layers: dict[int, int] | None = None

    @staticmethod
    def _preprocess_stim_circuit(circuit: stim.Circuit) -> tuple[stim.TableauSimulator, list[int]]:
        """Prepares a stim TableauSimulator for a given circuit and identifies measurement qubits.

        Args:
            circuit: The `stim.Circuit` to be processed.

        Returns:
            A tuple containing:
            - A `stim.TableauSimulator` with the circuit's unitary operations applied.
            - A list of qubit indices that are measured.
        """
        sim = stim.TableauSimulator()
        measurement_qubits = []
        for instruction in circuit.without_noise().flattened():
            if instruction.name == "M":
                measurement_qubits.extend([t.qubit_value for t in instruction.targets_copy()])
            else:
                sim.do(instruction)
        return sim, measurement_qubits

    @staticmethod
    def _is_output_possible(
        prepared_sim: stim.TableauSimulator, measurement_qubits: list[int], output: list[bool]
    ) -> bool:
        """Checks if a given measurement output is possible for a prepared stabilizer state.

        Args:
            prepared_sim: A `stim.TableauSimulator` in the state just before measurement.
            measurement_qubits: The list of qubits that were measured.
            output: A list of measurement outcomes (0 or 1, or False or True).

        Returns:
            True if the output is a possible outcome of the measurement, False otherwise.
        """
        sim_copy = prepared_sim.copy()
        for qubit, desired in zip(measurement_qubits, output):
            try:
                sim_copy.postselect_z(qubit, desired_value=desired)
            except ValueError:
                return False
        return True

    def _analyze(self) -> None:
        """Analyzes experiment results to compute the probability of valid outcomes.

        Returns:
            A dictionary mapping each qubit count to a list of scores (probabilities).
        """

        scores: dict[int, list[float]] = defaultdict(list)
        eff_layers: dict[int, list[float]] = defaultdict(list)
        classical_layers: dict[int, int] = {}

        for i, measurements in enumerate(tqdm(self.measurements, desc="Analyzing Circuits")):
            n_qubits = self.n_qubits_map[i]
            circuit = self.circuits[i]

            stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(circuit)
            sim, meas_qubits = self._preprocess_stim_circuit(stim_circuit)

            score = np.mean([self._is_output_possible(sim, meas_qubits, m) for m in measurements])
            scores[n_qubits].append(float(score))
            if score > 0:
                eff_layers[n_qubits].append(4 / float(score))

        for num_qubits in self.total_dof.keys():
            classical_layers[num_qubits] = math.ceil(np.log2(self.total_dof[num_qubits]))

        print("Analysis complete.")
        self._probability_of_valid_outcomes = scores
        self._effective_layers = eff_layers
        self._classical_layers = classical_layers

    @property
    def probability_of_valid_outcomes(self) -> dict[int, list[float]] | None:
        """Average fidelity of each circuit."""
        if self._probability_of_valid_outcomes is None:
            self._analyze()
        return self._probability_of_valid_outcomes

    @property
    def effective_layers(self) -> dict[int, list[float]] | None:
        """Effective number of layers required to solve the HLF problem for each circuit."""
        if self._effective_layers is None:
            self._analyze()
        return self._effective_layers

    @property
    def classical_layers(self) -> dict[int, int] | None:
        """Lower bound on number of classical layers required to solve the HLF problem."""
        if self._classical_layers is None:
            self._analyze()
        return self._classical_layers
