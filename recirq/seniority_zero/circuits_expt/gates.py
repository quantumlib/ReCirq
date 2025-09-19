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

"""Explicit decompositions of gates used in this experiment"""

import cirq
import numpy as np


def gsgate_from_sqrt_swap(q0: cirq.Qid, q1:cirq.Qid, theta: float) -> cirq.Circuit:
    """GS(theta) is a product of a Givens rotation (by theta) and a swap"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, z_exponent=-0.25, axis_phase_exponent=0).on(qubit)
            for qubit in [q0, q1]
        ]
    )
    circuit.moments.append(moment_1q1)
    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit.moments.append(moment_2q)
    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(
                x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25 - theta / np.pi
            ).on(q0),
            cirq.PhasedXZGate(
                x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25 + theta / np.pi
            ).on(q1),
        ]
    )
    circuit.moments.append(moment_1q2)
    circuit.moments.append(moment_2q)
    return circuit


def gsgate_from_cz(q0: cirq.Qid, q1: cirq.Qid, theta: float) -> cirq.Circuit:
    """GS(theta) is a product of a Givens rotation (by theta) and a swap"""
    circuit = cirq.Circuit()
    if theta > 0:
        theta -= np.pi
        moment_1q1 = cirq.Moment(
            [
                cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=1).on(q0),
                cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=1).on(q1),
            ]
        )
    else:
        moment_1q1 = cirq.Moment(
            [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q0)]
        )
    circuit.moments.append(moment_1q1)
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    circuit.moments.append(moment_2q)

    x_exponent23 = theta / np.pi + 0.5
    axis_phase_exponent23 = -0.5
    if x_exponent23 < 0:
        x_exponent23 = -x_exponent23
        axis_phase_exponent23 = axis_phase_exponent23 + 1

    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(
                x_exponent=x_exponent23, axis_phase_exponent=axis_phase_exponent23, z_exponent=1
            ).on(q1),
        ]
    )
    circuit.moments.append(moment_1q2)
    circuit.moments.append(moment_2q)

    moment_1q3 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(
                x_exponent=x_exponent23, axis_phase_exponent=axis_phase_exponent23, z_exponent=1
            ).on(q1),
        ]
    )
    circuit.moments.append(moment_1q3)
    circuit.moments.append(moment_2q)
    moment_1q4 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q0)]
    )
    circuit.moments.append(moment_1q4)
    return circuit


def cnot_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of a cnot from a cz to preserve 1q-2q layers"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q1)]
    )
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    moment_1q3 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)]
    )
    circuit._moments += [moment_1q1, moment_2q, moment_1q3]
    return circuit


def cnot_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of a cnot from sqrt_swap to preserve 1q-2q layers"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.25, axis_phase_exponent=-0.5).on(q1),
        ]
    )
    circuit._moments.append(moment_1q1)

    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit._moments.append(moment_2q)

    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25).on(q1),
        ]
    )
    circuit._moments.append(moment_1q2)
    circuit._moments.append(moment_2q)

    moment_1q3 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)]
    )
    circuit._moments.append(moment_1q3)
    return circuit


def cmnot_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """CMNOT is a NOT controlled by the |0> state of the control qubit"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, z_exponent=-0.75, axis_phase_exponent=-0.5).on(q1),
        ]
    )
    circuit._moments.append(moment_1q1)

    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit._moments.append(moment_2q)

    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25).on(q1),
        ]
    )
    circuit._moments.append(moment_1q2)
    circuit._moments.append(moment_2q)

    moment_1q3 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)]
    )
    circuit._moments.append(moment_1q3)
    return circuit


def cmnot_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """CMNOT is a NOT controlled by the |0> state of the control qubit"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=1).on(q1)]
    )
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    moment_1q3 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)]
    )
    circuit._moments += [moment_1q1, moment_2q, moment_1q3]
    return circuit


def swap_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of a swap gate from sqrt swap to preserve 1q-2q moment structure"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, z_exponent=-0.25, axis_phase_exponent=0).on(q)
            for q in [q0, q1]
        ]
    )
    circuit._moments.append(moment_1q1)

    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit._moments.append(moment_2q)
    circuit._moments.append(moment_1q1)
    circuit._moments.append(moment_2q)

    return circuit


def swap_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of a swap gate from cz to preserve 1q-2q moment structure"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q1)]
    )
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1),
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q0),
        ]
    )
    moment_1q3 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q1),
        ]
    )
    moment_1q4 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)]
    )

    circuit._moments += [
        moment_1q1,
        moment_2q,
        moment_1q2,
        moment_2q,
        moment_1q3,
        moment_2q,
        moment_1q4,
    ]
    return circuit


def swap_diagonalization_gate_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of a swap-diag gate from cz to preserve 1q-2q moment structure"""
    return gsgate_from_cz(q0, q1, np.pi / 4)


def swap_diagonalization_gate_from_sqrtswap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of a swap-diag gate from sqrt swap to preserve 1q-2q moment structure"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, z_exponent=0.25, axis_phase_exponent=0).on(q0),
            cirq.PhasedXZGate(x_exponent=0, z_exponent=-0.25, axis_phase_exponent=0).on(q1),
        ]
    )
    circuit.moments.append(moment_1q1)
    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit.moments.append(moment_2q)

    moment_1q2 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0, z_exponent=-0.5, axis_phase_exponent=0).on(q0)]
    )
    circuit.moments.append(moment_1q2)
    return circuit


def xxyy_diag_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Fixed construction of an XXYY-diag gate from cz to preserve 1q-2q moment structure"""
    return gsgate_from_cz(q0, q1, np.pi / 4)


def xxyy_diag_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Rotates 0.5*(XX+YY+IZ+ZI) -> ZI and 0.5*(IZ+ZI-XX-YY) -> IZ"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.5).on(q1)]
    )
    circuit.moments.append(moment_1q1)
    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit.moments.append(moment_2q)
    return circuit


def rzz_from_cz(q0: cirq.Qid, q1: cirq.Qid, theta: float) -> cirq.Circuit:
    """Rotates by exp(i*theta*ZZ) using a CZ gate"""
    moment_1q1 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q1)]
    )
    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=2 * theta / np.pi, axis_phase_exponent=0, z_exponent=0).on(
                q1
            )
        ]
    )
    moment_1q3 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1)]
    )
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    circuit = cirq.Circuit()
    circuit._moments += [moment_1q1, moment_2q, moment_1q2, moment_2q, moment_1q3]
    return circuit


def rzz_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid, theta: float)-> cirq.Circuit:
    raise NotImplementedError


def yyzz_diag_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Rotates YY -> ZI and ZZ -> IZ"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0, z_exponent=-0.5).on(q1),
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=0.5).on(q0),
        ]
    )
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    moment_1q3 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1),
        ]
    )
    circuit._moments += [moment_1q1, moment_2q, moment_1q3]
    return circuit


def xxzz_diag_from_cz(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Rotates XX -> ZI and ZZ -> IZ"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q1)]
    )
    moment_2q = cirq.Moment([cirq.CZ(q0, q1)])
    moment_1q3 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=0).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1),
        ]
    )
    circuit._moments += [moment_1q1, moment_2q, moment_1q3]
    return circuit


def yyzz_diag_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Rotates YY -> ZI and ZZ -> IZ"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.75, axis_phase_exponent=-1).on(q1),
        ]
    )
    circuit._moments.append(moment_1q1)

    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit._moments.append(moment_2q)

    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25).on(q1),
        ]
    )
    circuit._moments.append(moment_1q2)
    circuit._moments.append(moment_2q)

    moment_1q3 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=-1).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1),
        ]
    )
    circuit._moments.append(moment_1q3)
    return circuit


def xxzz_diag_from_sqrt_swap(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Rotates XX -> ZI and ZZ -> IZ"""
    circuit = cirq.Circuit()
    moment_1q1 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.25, axis_phase_exponent=-0.5).on(q1),
        ]
    )
    circuit._moments.append(moment_1q1)

    moment_2q = cirq.Moment([cirq.FSimGate(-np.pi / 4, np.pi / 2).on(q0, q1)])
    circuit._moments.append(moment_2q)

    moment_1q2 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=0.75).on(q0),
            cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=-0.25).on(q1),
        ]
    )
    circuit._moments.append(moment_1q2)
    circuit._moments.append(moment_2q)

    moment_1q3 = cirq.Moment(
        [
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=-0.5, z_exponent=-1).on(q0),
            cirq.PhasedXZGate(x_exponent=0.5, axis_phase_exponent=0.5, z_exponent=0).on(q1),
        ]
    )
    circuit._moments.append(moment_1q3)
    return circuit
