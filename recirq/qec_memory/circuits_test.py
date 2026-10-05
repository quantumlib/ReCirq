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

import cirq
import numpy as np

import recirq.qec_memory.circuits as qec_circuits


def test_replace_loop_repetitions():
    circuit = cirq.Circuit(
        cirq.X(cirq.q(0)),
        cirq.CircuitOperation(cirq.Circuit(cirq.X(cirq.q(0))).freeze(), repetitions=8),
        cirq.X(cirq.q(0)),
    )
    circuit2 = qec_circuits.replace_loop_repetitions(
        circuit, cycles=20, original_cycles=10
    )
    assert circuit2 == cirq.Circuit(
        cirq.X(cirq.q(0)),
        cirq.CircuitOperation(cirq.Circuit(cirq.X(cirq.q(0))).freeze(), repetitions=18),
        cirq.X(cirq.q(0)),
    )


def test_add_sweep_bits():
    qubits = cirq.GridQubit.rect(1, 10)
    measure_qubits = set(qubits[0::2])
    data_qubits = set(qubits[1::2])
    circuit = cirq.Circuit(*[cirq.M(q) for q in measure_qubits], cirq.M(*qubits))
    assert qec_circuits.identify_data_qubits(circuit) == data_qubits
    rng = np.random.default_rng(0)
    new_circuit = qec_circuits.add_sweep_bits(circuit, rng=rng)
    assert new_circuit == cirq.Circuit(
        cirq.X.on_each(cirq.q(0, 1), cirq.q(0, 9)) + circuit
    )
