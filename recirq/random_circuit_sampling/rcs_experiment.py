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

"""Tools for Random Circuit Sampling (RCS) and XEB analysis.

This module provides an experimental framework to generate, execute, and analyze
Random Circuit Sampling experiments across disjoint qubit patches. It supports
automated PhasedFSim characterization to determine hardware
gates for more accurate fidelity estimation.

The fidelity is calculated using Linear Cross-Entropy Benchmarking (XEB), as
detailed in:
    "Quantum supremacy using a programmable superconducting processor"
    Nature 563, 505–510 (2019).
    https://www.nature.com/articles/s41586-019-1666-5

Main components:
    - RCSExperiment: Manages parallel execution of circuit instances across
        disjoint grid patches.
    - RCSResults: Performs noiseless simulation and computes Linear XEB fidelity.
"""

import random
from collections import defaultdict
from typing import Dict, Tuple, List, Callable, Optional

import cirq
import cirq_google
import numpy as np
import networkx as nx
import cirq.contrib.routing as ccr
from tqdm import tqdm

import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments import xeb_fitting
import cirq.experiments.z_phase_calibration as xeb_characterize

def characterize_pairs(
    sampler: cirq.Sampler,
    qubits: List[cirq.GridQubit],
    gate: cirq.Gate,
    theta: bool = True,
    phi: bool = True,
    zeta: bool = True,
    chi: bool = True,
    gamma: bool = True
) -> Dict[Tuple[cirq.Qid, cirq.Qid], cirq.PhasedFSimGate]:
    """Performs XEB characterization for PhasedFSimGate angles.

    Args:
        sampler: The cirq.Sampler used for data collection.
        qubits: All qubits involved in the characterization.
        gate: The baseline ideal gate to calibrate against, e.g., cirq_google.SYC
        theta, phi, zeta, chi, gamma: Toggles for parameters to characterize.
            Defaults to True for all.

    Returns:
        A dictionary mapping qubit pairs to their calibrated PhasedFSimGate.
    """
    options = xeb_fitting.XEBPhasedFSimCharacterizationOptions(
        characterize_theta=theta,
        characterize_phi=phi,
        characterize_zeta=zeta,
        characterize_chi=chi,
        characterize_gamma=gamma
    ).with_defaults_from_gate(gate)

    results = xeb_characterize.calibrate_z_phases(
        sampler=sampler,
        qubits=qubits,
        options=options,
    )

    return results


def get_calibrated_circuit(circuit: cirq.Circuit, characterization: Optional[Dict] = None) -> cirq.Circuit:
    """Replaces ideal 2-qubit gates with calibrated PhasedFSimGate models.

    Args:
        circuit: The ideal RCS circuit to be updated.
        characterization: A mapping from qubit pairs (edges) to their
            measured PhasedFSimGate parameters.

    Returns:
        A new cirq.Circuit with gates updated to reflect hardware characterization.
    """
    if not characterization:
        return circuit

    def map_func(op: cirq.Operation, _):
        if len(op.qubits) == 2 and not cirq.is_measurement(op):
            edge = tuple(op.qubits)
            gate = characterization.get(edge) or characterization.get(edge[::-1])
            if gate: return gate.on(*op.qubits)
        return op

    return cirq.map_operations(circuit, map_func)


def make_rcs_circuit(
    qubits: List[cirq.GridQubit],
    depth: int,
    seed: Optional[int] = None,
    pattern_name: str = "staggered",
    single_qubit_gates: Tuple[cirq.Gate, ...] = (
        cirq.X ** 0.5,
        cirq.Y ** 0.5,
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25)
    ),
    two_qubit_op_factory: Callable[
        [cirq.GridQubit, cirq.GridQubit, np.random.RandomState],
        cirq.OP_TREE
    ] = lambda a, b, _: cirq_google.SYC(a, b)
) -> cirq.Circuit:
    """Generates a customizable RCS circuit using official Cirq experiment utilities.

    Args:
        qubits: The list of GridQubits to include in the circuit.
        depth: The number of interaction layers (cycles).
        seed: Random seed for gate selection and tiling reproducibility.
        pattern_name: Tiling configuration: "staggered", "half", or "aligned".
        single_qubit_gates: Collection of gates for random 1Q rotations.
        two_qubit_op_factory: Callable returning the 2Q operation for an edge.

    Returns:
        An cirq.Circuit instance.

    Raises:
        ValueError: If depth is negative or pattern_name is unrecognized.
    """
    if depth < 0:
        raise ValueError(f"Depth must be non-negative. Received: {depth}")

    patterns = {
        "staggered": cirq.experiments.GRID_STAGGERED_PATTERN,
        "half": cirq.experiments.HALF_GRID_STAGGERED_PATTERN,
        "aligned": cirq.experiments.GRID_ALIGNED_PATTERN,
    }

    if pattern_name not in patterns:
        raise ValueError(
            f"Invalid pattern_name '{pattern_name}'. Supported: {list(patterns.keys())}")

    circuit = rqcg.random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits,
        depth=depth,
        two_qubit_op_factory=two_qubit_op_factory,
        pattern=patterns[pattern_name],
        single_qubit_gates=single_qubit_gates,
        add_final_single_qubit_layer=True,
        seed=seed
    )

    return circuit


class RCSExperiment:
    """Manages generation and parallel execution of RCS experiments."""

    def __init__(
        self,
        patches: List[List[cirq.GridQubit]],
        depths: List[int],
        num_instances: int,
        pattern_name: str = "staggered",
        single_qubit_gates: Optional[Tuple[cirq.Gate, ...]] = None,
        two_qubit_gate: cirq.Gate = cirq_google.SYC,
        seed: Optional[int] = None
    ):
        """Initializes experiment and validates patch disjointness and connectivity.

        Args:
            patches: List of qubit lists representing disjoint sub-regions.
            depths: Cycle depths to execute.
            num_instances: Number of random circuit instances per (patch, depth).
            pattern_name: Tiling configuration, "staggered", "half", or "aligned".
            single_qubit_gates: Optional tuple of 1Q gates to use for rotations.
            two_qubit_gate: The ideal entangling gate.
            seed: Master seed for reproducibility.

        Raises:
            ValueError: If patches are not disjoint, contain isolated qubits,
                or are split into islands.
        """

        # Basic Disjointness Check
        all_qubits = [q for patch in patches for q in patch]
        if len(all_qubits) != len(set(all_qubits)):
            raise ValueError("Patches must be disjoint (no shared qubits).")

        # Patch Connectivity and Isolated Qubit Check
        for i, patch in enumerate(patches):
            patch_graph = ccr.gridqubits_to_graph_device(patch)

            # If a qubit is isolated, gridqubits_to_graph_device won't include it.
            if len(patch_graph.nodes) != len(patch):
                missing = set(patch) - set(patch_graph.nodes)
                raise ValueError(f"Patch {i} has isolated (dangling) qubits: {missing}")

            # Check for islands
            if not nx.is_connected(patch_graph):
                raise ValueError(f"Patch {i} is not connected (it is split into islands).")

        self.patches = patches
        self.all_qubits = all_qubits
        self.depths = depths
        self.num_instances = num_instances
        self.pattern_name = pattern_name
        self.two_qubit_gate = two_qubit_gate
        self.seed = seed

        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.single_qubit_gates = single_qubit_gates or (
            cirq.X ** 0.5, cirq.Y ** 0.5, cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25)
        )

    def run(
        self,
        sampler: cirq.Sampler,
        n_repetitions: int,
        characterize: bool = False,
        theta: bool = True,
        phi: bool = True,
        zeta: bool = True,
        chi: bool = True,
        gamma: bool = True
    ) -> "RCSResults":
        """Executes circuits in parallel using unique keys for patch separation.

        Args:
            sampler: Sampler to execute circuits.
            n_repetitions: Shots per circuit.
            characterize: If True, performs gate characterization before analysis.
            theta, phi, zeta, chi, gamma: Toggles for characterization angles.
                Default to True for all.

        Returns:
            An RCSResults object.
        """

        char_data = None
        if characterize:
            char_data = characterize_pairs(
                sampler=sampler,
                qubits=self.all_qubits,
                gate=self.two_qubit_gate,
                theta=theta,
                phi=phi,
                zeta=zeta,
                chi=chi,
                gamma=gamma
            )

        zipped_circuits = []
        metadata_flat = []
        all_individual_circuits = []

        for depth in self.depths:
            for instance_idx in range(self.num_instances):
                master_instance_seed = self._rng.randint(0, 2 ** 32 - 1)

                circuits_to_zip = []

                for patch_idx, qubits in enumerate(self.patches):
                    patch_seed = hash((master_instance_seed, patch_idx)) % (2 ** 32)

                    patch_circuit = make_rcs_circuit(
                        qubits=qubits,
                        depth=depth,
                        seed=patch_seed,
                        pattern_name=self.pattern_name,
                        single_qubit_gates=self.single_qubit_gates,
                        two_qubit_op_factory=lambda a, b, _: self.two_qubit_gate(a, b)
                    )

                    key = f"m_{patch_idx}"
                    patch_circuit.append(cirq.measure(*sorted(qubits), key=key))

                    circuits_to_zip.append(patch_circuit)
                    all_individual_circuits.append(patch_circuit)
                    metadata_flat.append({
                        "patch_idx": patch_idx,
                        "depth": depth,
                        "instance": instance_idx,
                        "key": key
                    })

                zipped_circuits.append(cirq.Circuit.zip(*circuits_to_zip))

        print(
            f"Running {len(zipped_circuits)} zipped instances on {len(self.patches)} parallel patches...")
        batch_results = sampler.run_batch(zipped_circuits, repetitions=n_repetitions)

        measurements_flat = []
        for i, combined_result in enumerate(batch_results):
            for patch_idx in range(len(self.patches)):
                key = f"m_{patch_idx}"
                patch_data = combined_result[0].measurements[key]
                measurements_flat.append(patch_data)

        return RCSResults(
            circuits=all_individual_circuits,
            measurements=measurements_flat,
            metadata=metadata_flat,
            characterization=char_data
        )


class RCSResults:
    """Calculates Linear Cross-Entropy Benchmarking (XEB) fidelity.

    Attributes:
        circuits: Unzipped individual circuits for the measurements.
        measurements: Measured bitstrings per circuit.
        metadata: Tracking info (e.g., patch, depth).
        characterization: A dictionary mapping qubit pairs to their calibrated PhasedFSimGate.
    """

    def __init__(
        self,
        circuits: List[cirq.Circuit],
        measurements: List[np.ndarray],
        metadata: List[Dict],
        characterization: Optional[Dict] = None
    ):
        self.circuits = circuits
        self.measurements = measurements
        self.metadata = metadata
        self.characterization = characterization
        self._fidelities_lin: Optional[Dict[Tuple[int, int], float]] = None

    def fidelities_lin(self, simulator: cirq.SimulatesAmplitudes = cirq.Simulator()) -> Dict:
        """Accesses or calculates Linear XEB fidelities.

        Args:
            simulator: The simulator to use for computing ideal amplitudes.  Defaults to
            cirq.Simulator().
        """
        if self._fidelities_lin is None:
            self._fidelities_lin = self._analyze(simulator=simulator)
        return self._fidelities_lin

    def _analyze(self, simulator: cirq.SimulatesAmplitudes) -> Dict[Tuple[int, int], float]:
        """Calculates linear xeb by comparing measured counts against ideal probabilities.

        Formula used for linear XEB: F = dimensions * E[P_ideal] - 1
        """
        fidelities = defaultdict(list)

        for i, circuit_measurements in enumerate(tqdm(self.measurements, desc="XEB Analysis")):
            raw_circuit = self.circuits[i]

            sim_circuit = get_calibrated_circuit(raw_circuit, self.characterization)

            program = cirq.Circuit(op for op in sim_circuit if not cirq.is_measurement(op))

            meta = self.metadata[i]
            patch_idx, depth = meta["patch_idx"], meta["depth"]
            qubits = sorted(raw_circuit.all_qubits())
            n_qubits = len(qubits)
            dim = 2 ** n_qubits

            # Convert bitstrings and calculate observed probabilities
            measured_ints = [cirq.big_endian_bits_to_int(m) for m in circuit_measurements]
            unique_ints, counts = np.unique(measured_ints, return_counts=True)
            measured_probs = counts / len(measured_ints)
            amplitudes = simulator.compute_amplitudes(
                program=program,
                bitstrings=list(unique_ints),
                qubit_order=qubits
            )
            ideal_probs = np.abs(np.array(amplitudes)) ** 2

            # Calculate linear fidelity
            linear_fidelity =  dim * (measured_probs @ ideal_probs) - 1
            fidelities[(patch_idx, depth)].append(linear_fidelity)

        return dict(fidelities)
