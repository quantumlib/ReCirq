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

import pytest
from recirq.qml_lfe import circuit_blocks
import cirq
import cirq_google
import numpy as np

_qubits = cirq.LineQubit.range(2)


@pytest.mark.parametrize(
    "known_circuit, compiled_ops",
    (
        ([cirq.SWAP(*_qubits)], circuit_blocks.swap_block(_qubits)),
        (
            [cirq.H(_qubits[0]), cirq.CNOT(*_qubits)],
            circuit_blocks.bell_pair_block(_qubits),
        ),
        (
            [cirq.CNOT(*_qubits), cirq.H(_qubits[0])],
            circuit_blocks.un_bell_pair_block(_qubits),
        ),
        (
            [
                cirq.X(_qubits[0]) ** 0.5,
                cirq.PhasedXZGate(
                    axis_phase_exponent=0.25, x_exponent=0.5, z_exponent=0
                )(_qubits[1]),
                cirq_google.SycamoreGate()(*_qubits),
            ],
            circuit_blocks.scrambling_block(_qubits, [0, 3]),
        ),
        (
            [
                cirq.Y(_qubits[0]) ** 0.5,
                cirq.Y(_qubits[1]) ** 0.5,
                cirq_google.SycamoreGate()(*_qubits),
            ],
            circuit_blocks.scrambling_block(_qubits, [2, 2]),
        ),
    ),
)
def test_known_blocks_equal(known_circuit, compiled_ops):
    desired_u = cirq.Circuit(known_circuit).unitary(dtype=np.complex128)
    actual_u = cirq.Circuit(compiled_ops).unitary(dtype=np.complex128)
    assert cirq.equal_up_to_global_phase(actual_u, desired_u)


def test_tsym_block_real():
    tsym_circuit = circuit_blocks.tsym_block(_qubits, [0, 0])  # no rotations.
    tsym_u = cirq.unitary(cirq.Circuit(tsym_circuit))
    assert np.all(tsym_u.imag < 1e-6)


def test_block_1d_circuit():
    depth = 8
    n_qubits = 11
    qubits = cirq.LineQubit.range(n_qubits)

    def _simple_fun(pairs, unused):
        assert len(unused) == 1
        return [cirq.CNOT(*pairs).with_tags(str(unused[0]))]

    test_block_circuit = circuit_blocks.block_1d_circuit(
        qubits, depth, _simple_fun, np.vstack(np.arange((depth * len(qubits) // 2)))
    )
    assert len(test_block_circuit) == depth
    assert len(test_block_circuit.all_qubits()) == n_qubits

    tot_i = 0
    for i, mom in enumerate(test_block_circuit):
        for op in mom:
            assert isinstance(op.gate, type(cirq.CNOT))
            assert op.tags[0] == str(tot_i)
            tot_i += 1

        # Number of operations and working depth will
        # always have parity that disagrees if number of
        # qubits is odd.
        assert i % 2 != tot_i % 2


def test_z_basis_gate():
    assert circuit_blocks.inv_z_basis_gate("Z") == cirq.I
    assert circuit_blocks.inv_z_basis_gate("X") == cirq.H
    assert circuit_blocks.inv_z_basis_gate("Y") == cirq.PhasedXZGate(
        axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-0.5
    )
