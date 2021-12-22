# Copyright 2021 Google
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

def test_DTCTask():
    np.random.seed(5)
    qubit_locations = [(3, 9), (3, 8), (3, 7), (4, 7), (4, 8), (5, 8), (5, 7), (5, 6), (6, 6), (6, 5), (7, 5), (8, 5),
                          (8, 4), (8, 3), (7, 3), (6, 3)]

    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    g = 0.94
    instances = 36
    initial_state = np.random.choice(2, num_qubits)
    local_fields = np.random.uniform(-1.0, 1.0, (instances, num_qubits))
    thetas = np.zeros((instances, num_qubits - 1))
    zetas = np.zeros((instances, num_qubits - 1))
    chis = np.zeros((instances, num_qubits - 1))
    gammas = -np.random.uniform(0.5*np.pi, 1.5*np.pi, (instances, num_qubits - 1))
    phis = -2*gammas
    args = ['qubits', 'g', 'initial_state', 'local_fields', 'thetas', 'zetas', 'chis', 'gammas', 'phis']
    default_resolvers = time_crystals.DTCTask().param_resolvers()
    for arg in args:
        kwargs = {}
        for name in args:
            kwargs[name] = None if name is arg else locals()[name]
        dtctask = time_crystals.DTCTask(disorder_instances=instances, **kwargs)
        param_resolvers = dtctask.param_resolvers()

def test_CompareDTCTask():
    np.random.seed(5)
    qubit_locations = [(3, 9), (3, 8), (3, 7), (4, 7), (4, 8), (5, 8)]#, (5, 7), (5, 6), (6, 6), (6, 5), (7, 5), (8, 5), (8, 4), (8, 3), (7, 3), (6, 3)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    disorder_instances = 8
    initial_state_instances = 6
    cycles = 10

    options = {
            'g': [0.6, 0.94],
            'local_fields': [np.random.uniform(-1.0, 1.0, (disorder_instances, num_qubits))],
            'initial_state': [[0]*num_qubits, [0,1]*(num_qubits//2), np.random.choice(2, num_qubits)],
            'initial_states': [np.random.choice(2, (disorder_instances, num_qubits))],
            'gammas': [np.random.uniform(-0.5*np.pi, -1.5*np.pi, (disorder_instances, num_qubits - 1))],
            'phis': [np.random.uniform(-1.5*np.pi, -0.5*np.pi, (disorder_instances, num_qubits - 1)), np.full((disorder_instances, num_qubits - 1), -0.4)],
            }
    for initial in ['initial_state', 'initial_states']:
        for variable in ['gammas', 'phis']:
            options_dict = {k:v for k,v in options.items() if k not in (initial, variable)}
            compare_dtctask = time_crystals.CompareDTCTask(qubits, cycles, disorder_instances, options_dict)
            for autocorrelate in [True,False]:
                for take_abs in [True,False]:
                    for index, polarizations in enumerate(time_crystals.run_comparison_experiment(compare_dtctask, autocorrelate, take_abs)):
                        print(index, autocorrelate, take_abs, initial, variable)
                        print(polarizations)
