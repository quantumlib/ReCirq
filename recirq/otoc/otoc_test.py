# Copyright 2020 Google
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

import cirq
import numpy as np

from recirq.otoc.otoc_circuits import build_otoc_circuits


# Simulate the normalization values of a 3-qubit OTOC circuit and ensure results are close to 1.
def test_build_otoc_circuits():
    np.random.seed(0)
    sampler = cirq.Simulator()
    qubit_locs = [(0, 1), (0, 2), (0, 3)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
    ancilla_loc = (0, 0)
    ancilla = cirq.GridQubit(*ancilla_loc)
    num_qubits = len(qubits)

    int_sets = [{(qubits[0], qubits[1])}, {(qubits[1], qubits[2])}]

    forward_ops = {
        (qubit_locs[i], qubit_locs[i + 1]): cirq.Circuit([cirq.CZ(qubits[i], qubits[i + 1])])
        for i in range(2)
    }

    reverse_ops = forward_ops

    cycles = range(4)
    rand_nums = np.random.choice(8, (num_qubits, max(cycles)))
    circuits = []
    for cycle in cycles:
        circs = build_otoc_circuits(
            qubits,
            ancilla,
            cycle,
            int_sets,
            forward_ops=forward_ops,
            reverse_ops=reverse_ops,
            butterfly_qubits=qubits[1],
            cycles_per_echo=2,
            sq_gates=rand_nums,
            use_physical_cz=True,
        )
        circuits.extend(circs.butterfly_I)

    results = []
    params = [{} for _ in range(len(circuits))]
    job = sampler.run_batch(programs=circuits, params_list=params, repetitions=2000)
    for d in range(len(circuits)):
        results.append(abs(np.mean(job[d][0].measurements["z"])))

    results = abs(2.0 * np.asarray(results) - 1.0)

    assert np.allclose(results, 1)
