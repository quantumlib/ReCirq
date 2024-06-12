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

import abc
import dataclasses
import hashlib
import pathlib
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import cirq
import numpy as np


# should go in Dedicated module
@dataclass(frozen=True, repr=False)
class Params(abc.ABC):
    name: str

    def __eq__(self, other):
        """A helper method to compare two dataclasses which might contain arrays."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        return all(
            array_compatible_eq(thing1, thing2)
            for thing1, thing2 in zip(
                dataclasses.astuple(self), dataclasses.astuple(other)
            )
        )

    @property
    def path_string(self) -> str:
        raise NotImplementedError()

    @property
    def base_path(self) -> pathlib.Path:
        return pathlib.Path(self.path_string + "_" + self.hash_key)

    @property
    def hash_key(self) -> str:
        """Gets the hash key for a set of params."""
        return hashlib.md5(str(self).encode("utf-8")).hexdigest()[0:16]

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        # This custom repr lets us add fields with default values without changing
        # the repr. This, in turn, lets us use the hash_key reliably even when
        # we add new fields with a default value.
        fields = dataclasses.fields(self)
        # adjusted_fields = [f for f in fields if getattr(self, f.name) != f.default]
        adjusted_fields = [
            f
            for f in fields
            if not array_compatible_eq(getattr(self, f.name), f.default)
        ]

        return (
            self.__class__.__qualname__
            + "("
            + ", ".join([f"{f.name}={getattr(self, f.name)}" for f in adjusted_fields])
            + ")"
        )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"


# Should go in dedicated module
@dataclass(frozen=True, eq=False)
class Data(abc.ABC):
    params: Params

    def __eq__(self, other):
        """A helper method to compare two dataclasses which might contain arrays."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        return all(
            array_compatible_eq(thing1, thing2)
            for thing1, thing2 in zip(
                dataclasses.astuple(self), dataclasses.astuple(other)
            )
        )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"


def array_compatible_eq(thing1, thing2):
    """A check for equality which can handle arrays."""
    if thing1 is thing2:
        return True

    # Here we handle dicts because they might have arrays in them.
    if isinstance(thing1, dict) and isinstance(thing2, dict):
        return all(
            array_compatible_eq(k_1, k_2) and array_compatible_eq(v_1, v_2)
            for (k_1, v_1), (k_2, v_2) in zip(thing1.items(), thing2.items())
        )
    if isinstance(thing1, np.ndarray) and isinstance(thing2, np.ndarray):
        return np.array_equal(thing1, thing2)
    if isinstance(thing1, np.ndarray) + isinstance(thing2, np.ndarray) == 1:
        return False
    try:
        return thing1 == thing2
    except TypeError:
        return NotImplemented


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
