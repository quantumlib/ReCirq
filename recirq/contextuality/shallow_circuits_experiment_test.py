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

"""Tests for the shallow circuits experiment for finding Hidden Linear Functions (HLF).

This test module verifies the core components of the HLF experiment, ensuring
that circuit generation, execution, and analysis perform as expected under
ideal (noiseless) conditions.
"""

import math

import cirq
import cirq_google
import numpy as np

import recirq.contextuality.shallow_circuits_experiment as hlf

DEVICE = cirq_google.Sycamore
ALL_QUBITS = list(DEVICE.metadata.qubit_set)
ALL_EDGES = list(DEVICE.metadata.nx_graph.edges)


def test_circuit_properties():
    """Tests properties of a single generated HLF circuit.

    This test verifies that the `hlf_circuit_biased_dropout` function
    generates a circuit with the expected structure and that the reported
    dropout fractions are accurate. It checks:
    - The total number of moments in the circuit.
    - The classical complexity (DOF) is above ideal quantum layers, four.
    - The actual gate dropout fractions are close to the requested values.
    - The relationship between the dropout fraction and the retention fraction holds.
    """

    # Test on large even number of qubits to have intended actual drop fraction for S and CZ gates
    num_qubits = [40, 42, 44, 46, 48, 50]
    s_dropout_fraction = 0.5
    cz_dropout_fraction = 0.5
    seed = 42

    for n_qubits in num_qubits:
        result_circuit = hlf.hlf_circuit_biased_dropout(
            n_qubits, ALL_QUBITS, ALL_EDGES, cz_dropout_fraction, s_dropout_fraction, seed
        )
        circuit = result_circuit.circuit
        total_dof = result_circuit.total_dof
        original_cz_layers = result_circuit.original_cz_layers
        final_cz_layers = result_circuit.final_cz_layers
        actual_cz_drop = result_circuit.actual_cz_dropout_fraction
        actual_s_drop = result_circuit.actual_s_dropout_fraction

        # Check the length of moments in the circuit
        assert len(circuit) == 8
        assert len(original_cz_layers) == 4
        assert len(final_cz_layers) == 4

        # Check the classical lower bound is greater than 4
        assert math.ceil(np.log2(total_dof)) > 4

        # Check the actual gate dropout fraction is close to what was asked
        assert np.isclose(cz_dropout_fraction, actual_cz_drop, atol=0.01)
        assert np.isclose(s_dropout_fraction, actual_s_drop, atol=0.01)
        assert np.isclose(
            1 - actual_cz_drop, len(sum(final_cz_layers, [])) / len(sum(original_cz_layers, []))
        )


def test_experiment():
    """Tests the full experiment workflow using a noiseless simulator.

    This test runs the entire experiment from circuit generation to analysis
    on a perfect simulator. It verifies that the results match theoretical
    expectations for a noiseless execution:
    - The probability of valid outcomes should be 1.0.
    - The effective number of layers should be 4.0.
    - The classical complexity should be greater than the effective layers.
    """

    n_qubits_list = np.arange(10, 15, 1)
    s_dropout_fraction = 0.5
    cz_dropout_fraction = 0.5
    n_runs = 100

    sampler = cirq.Simulator()
    n_repetitions = 10

    hlf_experiment = hlf.ShallowCircuitExperiment(
        all_qubits=ALL_QUBITS,
        all_edges=ALL_EDGES,
        n_qubits_list=n_qubits_list,
        s_dropout_fraction=s_dropout_fraction,
        cz_dropout_fraction=cz_dropout_fraction,
        n_runs=n_runs,
    )

    results = hlf_experiment.run(sampler, n_repetitions)

    probability_of_valid_outcomes = results.probability_of_valid_outcomes
    effective_number_of_layers = results.effective_layers
    classical_layers = results.classical_layers

    # Running noiseless experiment, mean should be 1
    for key in probability_of_valid_outcomes.keys():
        assert np.mean(probability_of_valid_outcomes[key]) == 1

    # Running noiseless experiment, mean should be 4
    for key in effective_number_of_layers.keys():
        assert np.mean(effective_number_of_layers[key]) == 4

    # Running noiseless experiment, classical layers should be more than the effective layers
    for key in classical_layers.keys():
        assert np.mean(effective_number_of_layers[key]) < classical_layers[key]
