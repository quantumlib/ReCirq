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
from typing import Sequence, Union, Optional

import cirq
import cirq_google as cg
import numpy as np
from cirq.optimizers.two_qubit_to_fsim import (
    _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops,
    _fix_single_qubit_gates_around_kak_interaction,
)


def _decompose_xx_yy(
    desired_interaction: Union[cirq.Operation, np.ndarray, "cirq.SupportsUnitary"],
    *,
    available_gate: cirq.Gate,
    atol: float = 1e-8,
    qubits: Optional[Sequence[cirq.Qid]] = None,
) -> cirq.Circuit:
    if qubits is None:
        if isinstance(desired_interaction, cirq.Operation):
            qubits = desired_interaction.qubits
        else:
            qubits = cirq.LineQubit.range(2)
    kak = cirq.kak_decomposition(desired_interaction)

    x, y, z = kak.interaction_coefficients
    if abs(z) > atol:
        raise ValueError(f"zz term present in {desired_interaction!r} -> {kak}")

    if isinstance(available_gate, cirq.ISwapPowGate):
        available_gate = cirq.FSimGate(-np.pi / 2 * available_gate.exponent, 0)

    if (
        min(x, y) > np.pi / 4 - atol
        and abs(available_gate.phi) < atol
        and abs(available_gate.theta - np.pi / 4) < atol
    ):
        ops = [
            available_gate(*qubits),
            cirq.Y.on_each(*qubits),
            available_gate(*qubits),
        ]
    elif (
        min(x, y) > np.pi / 4 - atol
        and abs(available_gate.phi) < atol
        and abs(available_gate.theta + np.pi / 4) < atol
    ):
        ops = [
            available_gate(*qubits),
            available_gate(*qubits),
        ]
    else:
        ops = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(
            qubits=qubits,
            fsim_gate=available_gate,
            canonical_x_kak_coefficient=x,
            canonical_y_kak_coefficient=y,
        )
    result = _fix_single_qubit_gates_around_kak_interaction(
        desired=kak,
        operations=ops,
        qubits=qubits,
    )
    output = cirq.Circuit(result)
    cirq.testing.assert_allclose_up_to_global_phase(
        output.unitary(qubit_order=qubits), cirq.unitary(desired_interaction), atol=1e-4
    )
    return output


def controlled_iswap(
    a: cirq.Qid, b: cirq.Qid, c: cirq.Qid, inverse: Optional[bool] = False
):
    """Performs ISWAP(a,b).controlled_by(c).

    Returns a generator of cirq.Operations.
    """
    available_gate = cirq.ISWAP**0.5
    if inverse:
        mul = -1
    else:
        mul = 1

    def convert_op(op: cirq.Operation) -> cirq.OP_TREE:
        if len(op.qubits) == 2:
            return _decompose_xx_yy(op, available_gate=available_gate)
        return op

    def simplify_op(op: cirq.Operation) -> cirq.Operation:
        if len(op.qubits) == 1:
            return cirq.PhasedXZGate.from_matrix(cirq.unitary(op)).on(*op.qubits)
        if op.gate == cirq.FSimGate(-np.pi / 4, 0):
            return cirq.ISWAP(*op.qubits) ** 0.5
        return op

    circuit = cirq.Circuit(
        cirq.CNOT(b, a),
        cirq.CNOT(c, b) ** (-0.5 * mul),
        cirq.CZ(a, b),
        cirq.CNOT(c, b) ** (0.5 * mul),
        cirq.Y(a).controlled_by(b),
        cirq.S(b) ** -1,
    )

    converted_ops = cirq.Circuit(convert_op(op) for op in circuit.all_operations())
    cirq.MergeSingleQubitGates().optimize_circuit(converted_ops)
    cirq.DropNegligible().optimize_circuit(converted_ops)
    return cirq.Circuit(simplify_op(op) for op in converted_ops.all_operations())


def controlled_sqrt_iswap(a: cirq.Qid, b: cirq.Qid, c: cirq.Qid):
    """Performs (ISWAP(a,b)i**0.5).controlled_by(c)"""
    return cg.optimized_for_sycamore(
        cirq.Circuit(
            cirq.CNOT(a, b),
            cirq.TOFFOLI(c, b, a) ** -0.5,
            cirq.CZ(c, b) ** 0.25,
            cirq.CNOT(a, b),
        )
    )


def controlled_inv_sqrt_iswap(a: cirq.Qid, b: cirq.Qid, c: cirq.Qid):
    """Performs (ISWAP(a,b)i**0.5).controlled_by(c)"""
    return cg.optimized_for_sycamore(
        cirq.Circuit(
            cirq.CNOT(a, b),
            cirq.TOFFOLI(c, b, a) ** 0.5,
            cirq.CZ(c, b) ** -0.25,
            cirq.CNOT(a, b),
        )
    )
