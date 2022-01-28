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
import cirq
import numpy as np
import itertools


def test_DTCExperiment():
    """Test to check all combinations of defaults vs supplied inputs for
    DTCExperiment, with the goal of checking all paths for crashes

    """
    np.random.seed(5)
    qubit_locations = [
        (3, 9),
        (3, 8),
        (3, 7),
        (4, 7),
        (4, 8),
        (5, 8),
        (5, 7),
        (5, 6),
        (6, 6),
        (6, 5),
        (7, 5),
        (8, 5),
        (8, 4),
        (8, 3),
        (7, 3),
        (6, 3),
    ]

    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    g = 0.94
    disorder_instances = 36
    initial_states = np.random.choice(2, (disorder_instances, num_qubits))
    local_fields = np.random.uniform(-1.0, 1.0, (disorder_instances, num_qubits))
    thetas = np.zeros((disorder_instances, num_qubits - 1))
    zetas = np.zeros((disorder_instances, num_qubits - 1))
    chis = np.zeros((disorder_instances, num_qubits - 1))
    gammas = -np.random.uniform(
        0.5 * np.pi, 1.5 * np.pi, (disorder_instances, num_qubits - 1)
    )
    phis = -2 * gammas
    arguments = [
        qubits,
        g,
        disorder_instances,
        initial_states,
        local_fields,
        thetas,
        zetas,
        chis,
        gammas,
        phis,
    ]
    argument_names = [
        "qubits",
        "g",
        "disorder_instances",
        "initial_states",
        "local_fields",
        "thetas",
        "zetas",
        "chis",
        "gammas",
        "phis",
    ]
    default_resolvers = time_crystals.DTCExperiment().param_resolvers()
    for arguments_instance in itertools.product(
        *zip(arguments, [None] * len(arguments))
    ):
        kwargs = {
            name: instance
            for (name, instance) in zip(argument_names, arguments_instance)
            if instance is not None
        }
        dtcexperiment = time_crystals.DTCExperiment(**kwargs)
        param_resolvers = list(dtcexperiment.param_resolvers())


def test_comparison_experiments():
    """Test to check all combinations of defaults vs supplied inputs for
    comparison_experiments, with the goal of checking all paths for crashes

    """
    np.random.seed(5)
    qubit_locations = [
        (3, 9),
        (3, 8),
        (3, 7),
        (4, 7),
        (4, 8),
        (5, 8),
        (5, 7),
        (5, 6),
        (6, 6),
        (6, 5),
        (7, 5),
        (8, 5),
        (8, 4),
        (8, 3),
        (7, 3),
        (6, 3),
    ]

    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    g_cases = [0.94, 0.6]
    disorder_instances = 36
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
    for argument_case in itertools.product(
        *zip([None] * len(argument_cases), argument_cases)
    ):
        kwargs = dict(zip(argument_names, argument_case))
        for experiment in time_crystals.comparison_experiments(
            qubits, disorder_instances, **kwargs
        ):
            param_resolvers = list(experiment.param_resolvers())
