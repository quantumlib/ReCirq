# Copyright 2023 Google
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

"""Specialized gate classes for this experiment"""
from typing import Generator, List

import cirq
import numpy as np


class GSGate(cirq.Gate):
    """Givens-swap gate: a product of a Givens rotation gate and a swap."""

    def __init__(self, angle: float):
        """Givens-swap gate: a product of Givens rotation gate and swap.

        Args:
            angle[float]: the angle to rotate by
        """
        super(GSGate, self)
        self.angle = angle

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.sin(self.angle), np.cos(self.angle), 0],
                [0, np.cos(self.angle), -np.sin(self.angle), 0],
                [0, 0, 0, 1],
            ]
        )

    def _decompose_(self, qubits: List[cirq.Qid]) -> Generator:
        q0, q1 = qubits
        yield cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q0)
        yield cirq.CZ(q0, q1)
        yield [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(
                x_exponent=self.angle / np.pi + 0.5, axis_phase_exponent=-0.5, z_exponent=1
            ).on(q1),
        ]
        yield cirq.CZ(q0, q1)
        yield [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(
                x_exponent=self.angle / np.pi + 0.5, axis_phase_exponent=-0.5, z_exponent=1
            ).on(q1),
        ]
        yield cirq.CZ(q0, q1)
        yield cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q0)

    def __pow__(self, exponent: float) -> cirq.Gate:
        # Gate is a reflection for all angles and thus self-inverse,
        # but a fractional power of a GSGate is not equal to another GSGate.
        if exponent == -1:
            return GSGate(self.angle)
        raise NotImplementedError

    def _circuit_diagram_info_(self, args) -> List[str]:
        return [f"GS({self.angle})"] * self.num_qubits()


class ZIpXYRotationGate(cirq.Gate):
    """A rotation exp[i * (-angle / 2) * (IZ + ZI + XX + YY)]"""

    def __init__(self, angle: float):
        """A rotation exp[i * (-angle / 2) * (IZ + ZI + XX + YY)]

        Args:
            angle[float]: the angle to rotate by
        """
        super(ZIpXYRotationGate, self)
        self.angle = angle

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [np.exp(-1j * self.angle / 2), 0, 0, 0],
                [0, np.cos(-self.angle / 2), 1j * np.sin(-self.angle / 2), 0],
                [0, 1j * np.sin(-self.angle / 2), np.cos(-self.angle / 2), 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ]
        )

    def _decompose_(self, qubits: List[cirq.Qid]) -> Generator:
        # This is suboptimal - it uses 6 CZ gates, while I think this
        # should be possible in less.
        yield GSGate(np.pi / 4).on(*qubits)
        yield cirq.rz(self.angle).on(qubits[0])
        yield GSGate(np.pi / 4).on(*qubits)

    def __pow__(self, exponent: float) -> cirq.Gate:
        return ZIpXYRotationGate(self.angle * exponent)

    def _circuit_diagram_info_(self, args) -> List[str]:
        return [f"ZIpXY({self.theta})"] * self.num_qubits()


class ZImXYRotationGate(cirq.Gate):
    """A rotation exp[i * (-angle / 2) * (IZ + ZI - (XX + YY))]"""

    def __init__(self, angle: float):
        """A rotation exp[i * (-angle / 2) * (IZ + ZI - (XX + YY))]

        Args:
            angle[float]: the angle to rotate by
        """
        super(ZImXYRotationGate, self)
        self.angle = angle

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [np.exp(-1j * self.angle / 2), 0, 0, 0],
                [0, np.cos(-self.angle / 2), -1j * np.sin(-self.angle / 2), 0],
                [0, -1j * np.sin(-self.angle / 2), np.cos(-self.angle / 2), 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ]
        )

    def _decompose_(self, qubits: List[cirq.Qid]) -> Generator:
        # This is suboptimal - it uses 6 CZ gates, while I think this
        # should be possible in less.
        yield GSGate(np.pi / 4).on(*qubits)
        yield cirq.rz(self.angle).on(qubits[1])
        yield GSGate(np.pi / 4).on(*qubits)

    def __pow__(self, exponent: float) -> cirq.Gate:
        return ZImXYRotationGate(self.angle * exponent)

    def _circuit_diagram_info_(self, args) -> List[str]:
        return [f"ZImXY({self.theta})"] * self.num_qubits()


class ZZRotationGate(cirq.Gate):
    """A rotation around ZZ"""

    def __init__(self, angle: float):
        """A rotation around ZZ

        Args:
            angle[float]: the angle to rotate by
        """
        super(ZZRotationGate, self)
        self.angle = angle

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [np.exp(1j * self.angle), 0, 0, 0],
                [0, np.exp(-1j * self.angle), 0, 0],
                [0, 0, np.exp(-1j * self.angle), 0],
                [0, 0, 0, np.exp(1j * self.angle)],
            ]
        )

    def _decompose_(self, qubits: List[cirq.Qid]) -> Generator:
        q0, q1 = qubits
        yield cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q1)
        yield cirq.CZ(q0, q1)
        yield cirq.PhasedXZGate(
            x_exponent=2 * self.angle / np.pi, axis_phase_exponent=0, z_exponent=0
        ).on(q1)
        yield cirq.CZ(q0, q1)
        yield cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)

    def __pow__(self, exponent: float) -> cirq.Gate:
        return ZZRotationGate(self.angle * exponent)

    def _circuit_diagram_info_(self, args) -> List[str]:
        return [f"rZZ({self.theta})"] * self.num_qubits()
