# Copyright 2024 Google
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

from typing import Optional, Sequence, Union

import cirq
import numpy as np


# DELETE?
class ConstantTwoQubitGateDepolarizingNoiseModel(cirq.NoiseModel):
    """Applies noise to each two qubit gate individually at the end of every moment."""

    def __init__(self, depolarizing_probability):
        self.noise_gate = cirq.DepolarizingChannel(depolarizing_probability)

    @property
    def name(self):
        return "noise_gate :" + repr(self.noise_gate) + "after each two qubit gate"

    def noisy_operation(self, operation: "cirq.Operation"):
        not_measurement = not isinstance(operation.gate, cirq.ops.MeasurementGate)
        if len(operation.qubits) > 1 and not_measurement:
            return [operation] + [self.noise_gate(q) for q in operation.qubits]
        else:
            return operation


# Move to experiment.py
def get_noise_model(
    name: str, params: Optional[Sequence[float]]
) -> Union[None, cirq.NoiseModel]:
    if name == "ConstantTwoQubitGateDepolarizingNoiseModel":
        assert params is not None
        assert len(params) == 1
        return ConstantTwoQubitGateDepolarizingNoiseModel(params[0])
    elif name == "None":
        return None
    else:
        raise NotImplementedError("Noise model not implemented")


# move to shadow tomography
def reorder_qubit_wavefunction(
    *,
    wf: np.ndarray,
    qubits_old_order: Sequence[cirq.Qid],
    qubits_new_order: Sequence[cirq.Qid],
) -> np.ndarray:
    """Reorders the amplitudes of wf based on permutation of qubits.

    May or may not destructively affect the original argument wf.
    """
    assert set(qubits_new_order) == set(qubits_old_order)
    n_qubits = len(qubits_new_order)
    assert wf.shape == (2**n_qubits,)

    wf = np.reshape(wf, (2,) * n_qubits)
    axes = tuple(qubits_old_order.index(qubit) for qubit in qubits_new_order)
    wf = np.transpose(wf, axes)
    wf = np.reshape(wf, (2**n_qubits,))

    return wf


# move to cirq_op or something
def is_expected_elementary_cirq_op(op: cirq.Operation) -> bool:
    """Checks whether op is one of the operations that we expect when decomposing a quaff op."""
    to_keep = isinstance(
        op.gate,
        (
            cirq.HPowGate,
            cirq.CXPowGate,
            cirq.ZPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.CZPowGate,
        ),
    )
    return to_keep
