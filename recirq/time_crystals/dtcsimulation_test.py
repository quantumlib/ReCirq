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

import recirq.time_crystals as time_crystals
import numpy as np
import itertools
from typing import List
import cirq


def test_run_comparison_experiment():
    """Test to check all combinations of defaults vs supplied inputs for
    run_comparison_experiments, with the goal of checking all paths for crashes

    """

    print("testing run_comparison_rexperiment")
    np.random.seed(5)
    qubit_locations = [
        (3, 9),
        (3, 8),
        (3, 7),
        (4, 7),
    ]

    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    cycles = 5
    g_cases = [0.94, 0.6]
    disorder_instances = 5
    initial_states_cases = [
        np.random.choice(2, (disorder_instances, num_qubits)),
        np.tile(np.random.choice(2, num_qubits), (disorder_instances, 1)),
    ]
    local_fields_cases = [
        np.random.uniform(-1.0, 1.0, (disorder_instances, num_qubits)),
        np.tile(np.random.uniform(-1.0, 1.0, num_qubits), (disorder_instances, 1)),
    ]
    phis_cases = [
        np.random.uniform(np.pi, 3 * np.pi, (disorder_instances, num_qubits - 1)),
        np.full((disorder_instances, num_qubits - 1), 0.4),
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
                qubits, cycles, disorder_instances, autocorrelate, take_abs, **kwargs
            ):
                pass
    print("testing run_comparison_rexperiment complete")


def test_signal_ratio():
    """Test signal_ratio function with random `np.ndarrays`"""

    print("testing signal_ratio")
    np.random.seed(5)
    cycles = 100
    num_qubits = 16
    zeta_1 = np.random.uniform(-1.0, 1.0, (cycles, num_qubits))
    zeta_2 = np.random.uniform(-1.0, 1.0, (cycles, num_qubits))
    res = time_crystals.signal_ratio(zeta_1, zeta_2)
    print("testing signal_ratio complete")


def test_symbolic_dtc_circuit_list():
    """Test symbolic_dtc_circuit_list function for select qubits and cycles"""

    print("testing symbolic_dtc_circuit_list")
    qubit_locations = [
        (3, 9),
        (3, 8),
        (3, 7),
        (4, 7),
    ]

    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    cycles = 5
    circuit_list = time_crystals.symbolic_dtc_circuit_list(qubits, cycles)
    print("testing symbolic_dtc_circuit_list complete")
