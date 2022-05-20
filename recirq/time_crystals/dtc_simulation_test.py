# Copyright 2022 Google
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

from typing import List, Tuple

import numpy as np
import itertools
import pytest

import cirq
import recirq.time_crystals as time_crystals

QUBIT_LOCATIONS = [
    (3, 9),
    (3, 8),
    (3, 7),
    (4, 7),
]

QUBITS = [cirq.GridQubit(*idx) for idx in QUBIT_LOCATIONS]
NUM_QUBITS = len(QUBITS)


def probabilities_predicate(probabilities: np.ndarray, shape: Tuple[int, int]) -> bool:
    return (
        probabilities.shape == shape
        and np.all(0 <= probabilities)
        and np.all(probabilities <= 1)
        and np.all(np.isclose(np.sum(probabilities, axis=1), 1))
    )


def polarizations_predicate(polarizations: np.ndarray, shape: Tuple[int, int]) -> bool:
    return (
        polarizations.shape == shape
        and np.all(-1 <= polarizations)
        and np.all(polarizations <= 1)
    )


def test_simulate_dtc_circuit_list_sweep() -> None:
    cycles = 5
    circuit_list = time_crystals.symbolic_dtc_circuit_list(QUBITS, cycles)
    param_resolvers = time_crystals.DTCExperiment(qubits=QUBITS).param_resolvers()
    qubit_order = QUBITS
    for probabilities in time_crystals.dtc_simulation.simulate_dtc_circuit_list_sweep(
        circuit_list, param_resolvers, qubit_order
    ):
        assert probabilities_predicate(probabilities, (cycles + 1, 2**NUM_QUBITS))

    with pytest.raises(
        ValueError, match="circuits in circuit_list are not in increasing order of size"
    ):
        time_crystals.dtc_simulation.simulate_dtc_circuit_list(
            list(reversed(circuit_list)), param_resolvers, qubit_order
        )


def test_simulate_dtc_circuit_list() -> None:
    cycles = 5
    circuit_list = time_crystals.symbolic_dtc_circuit_list(QUBITS, cycles)
    param_resolver = next(
        iter(time_crystals.DTCExperiment(qubits=QUBITS).param_resolvers())
    )
    qubit_order = QUBITS
    probabilities = time_crystals.dtc_simulation.simulate_dtc_circuit_list(
        circuit_list, param_resolver, qubit_order
    )
    assert probabilities_predicate(probabilities, (cycles + 1, 2**NUM_QUBITS))
    with pytest.raises(
        ValueError, match="circuits in circuit_list are not in increasing order of size"
    ):
        time_crystals.dtc_simulation.simulate_dtc_circuit_list(
            list(reversed(circuit_list)), param_resolver, qubit_order
        )


def test_get_polarizations() -> None:
    cycles = 5
    np.random.seed(5)
    probabilities = np.random.uniform(0, 1, (cycles, 2**NUM_QUBITS))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    initial_states = np.random.choice(2, NUM_QUBITS)
    assert polarizations_predicate(
        time_crystals.dtc_simulation.get_polarizations(probabilities, NUM_QUBITS),
        (cycles, NUM_QUBITS),
    )
    assert polarizations_predicate(
        time_crystals.dtc_simulation.get_polarizations(
            probabilities, NUM_QUBITS, initial_states
        ),
        (cycles, NUM_QUBITS),
    )


def test_simulate_for_polarizations() -> None:
    cycles = 5
    circuit_list = time_crystals.symbolic_dtc_circuit_list(qubits=QUBITS, cycles=cycles)
    dtcexperiment = time_crystals.DTCExperiment(qubits=QUBITS)
    for autocorrelate, take_abs in itertools.product([True, False], repeat=2):
        assert polarizations_predicate(
            time_crystals.dtc_simulation.simulate_for_polarizations(
                dtcexperiment=dtcexperiment,
                circuit_list=circuit_list,
                autocorrelate=autocorrelate,
                take_abs=take_abs,
            ),
            (cycles + 1, NUM_QUBITS),
        )


def test_run_comparison_experiment() -> None:
    """Test to check all combinations of defaults vs supplied inputs for
    run_comparison_experiments, with the goal of checking all paths for crashes

    """
    np.random.seed(5)
    cycles = 5
    g_cases = [0.94, 0.6]
    disorder_instances = 5
    initial_states_cases = [
        np.random.choice(2, (disorder_instances, NUM_QUBITS)),
        np.tile(np.random.choice(2, NUM_QUBITS), (disorder_instances, 1)),
    ]
    local_fields_cases = [
        np.random.uniform(-1.0, 1.0, (disorder_instances, NUM_QUBITS)),
        np.tile(np.random.uniform(-1.0, 1.0, NUM_QUBITS), (disorder_instances, 1)),
    ]
    phis_cases = [
        np.random.uniform(np.pi, 3 * np.pi, (disorder_instances, NUM_QUBITS - 1)),
        np.full((disorder_instances, NUM_QUBITS - 1), 0.4),
    ]
    argument_names = [
        "g_cases",
        "initial_states_cases",
        "local_fields_cases",
        "phis_cases",
    ]
    argument_cases = [g_cases, initial_states_cases, local_fields_cases, phis_cases]
    paired_with_none = zip(argument_cases, [None] * len(argument_cases))

    for autocorrelate, take_abs in itertools.product([True, False], repeat=2):
        for argument_case in itertools.product(*paired_with_none):
            named_arguments = zip(argument_names, argument_case)
            kwargs = {
                name: args for (name, args) in named_arguments if args is not None
            }
            for polarizations in time_crystals.run_comparison_experiment(
                QUBITS, cycles, disorder_instances, autocorrelate, take_abs, **kwargs
            ):
                assert polarizations_predicate(polarizations, (cycles + 1, NUM_QUBITS))


def test_signal_ratio() -> None:
    """Test signal_ratio function with random `np.ndarrays`"""
    np.random.seed(5)
    cycles = 100
    num_qubits = 16
    zeta_1 = np.random.uniform(-1.0, 1.0, (cycles, num_qubits))
    zeta_2 = np.random.uniform(-1.0, 1.0, (cycles, num_qubits))
    res = time_crystals.signal_ratio(zeta_1, zeta_2)
    assert np.all(res >= 0) and np.all(res <= 1)


def test_symbolic_dtc_circuit_list() -> None:
    """Test symbolic_dtc_circuit_list function for select qubits and cycles"""
    cycles = 5
    circuit_list = time_crystals.symbolic_dtc_circuit_list(QUBITS, cycles)
    for index, circuit in enumerate(circuit_list):
        assert len(circuit) == 3 * index + 1
