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

from dataclasses import dataclass
from typing import Callable

import numpy as np

import cirq
import cirq_google as cg


@dataclass(frozen=True)
class HomogeneousCircuitStats:
    """Statistics about a well-structured circuit.

    See Also:
        :py:func:`validate_well_structured`

    Attributes:
        num_phx: Number of layers of PhasedXPowGate
        num_z: Number of layers of ZPowGate
        num_syc: Number of layers of SYC entangling gates
        has_permutation: Whether there's a classical permutation of
            indices at the end of the circuit
        has_measurement: Whether there's a measurement at the end of the
            circuit
    """
    num_phx: int
    num_z: int
    num_syc: int
    has_permutation: bool = False
    has_measurement: bool = False


def _homogeneous_moment(
        moment: 'cirq.Moment',
        op_condition: Callable[[cirq.Operation], bool]) -> bool:
    cases = {op_condition(op) for op in moment}
    if len(cases) != 1:
        # Non-homogeneous
        return False
    return list(cases)[0]


def _homogeneous_gate_type(moment, gate_type):
    return _homogeneous_moment(
        moment,
        lambda op: op.gate is not None and isinstance(op.gate, gate_type))


MOMENT_GATE_CLASSES = [
    cirq.PhasedXPowGate,
    cirq.ZPowGate,
    cg.SycamoreGate,
    cirq.MeasurementGate,
    cirq.QubitPermutationGate,
]


def get_moment_class(moment: cirq.Moment):
    """For a given moment, return the Gate class which every operation
    is an instantiation or `None` for non-homogeneous or non-hardware moments.
    """
    mom_class_vec = [_homogeneous_gate_type(moment, gc) for gc in MOMENT_GATE_CLASSES]
    if np.sum(mom_class_vec) != 1:
        # Non-homogeneous or not in gcs
        return None

    return MOMENT_GATE_CLASSES[np.argmax(mom_class_vec)]


def get_moment_classes(circuit: cirq.Circuit):
    """Return the 'moment class' for each moment in the circuit.

    A moment class is the Gate class of which every operation
    is an instantiation or `None` for non-homogeneous or non-hardware moments.
    """
    return [get_moment_class(moment) for moment in circuit.moments]


def _find_circuit_structure_violations(mom_classes):
    violations = np.asarray([mom_class is None for mom_class in mom_classes])
    violation_indices = np.where(violations)[0]
    return violation_indices


def find_circuit_structure_violations(circuit: cirq.Circuit):
    """Return indices where the circuit contains non-homogenous
    non-hardware moments.
    """
    mom_classes = get_moment_classes(circuit)
    return _find_circuit_structure_violations(mom_classes)


class BadlyStructuredCircuitError(ValueError):
    pass


def validate_well_structured(circuit: cirq.Circuit,
                             allow_terminal_permutations=False):
    """Raises a ValueError if the circuit is not structured, otherwise returns
    a list of moment classes (see `get_moment_classes`) and a
    `HomogeneousCircuitStats` set of statistics.

    A "structured circuit" means that each moment contains either all-PhX,
    all-Z, all-SYC, or all-SQRT_ISWAP (i.e. moments have homogeneous gate types).
    The gate type common in a given moment can be called the "moment class".
    In structured circuits, moments are arranged so that the layers are always
    PhX, Z, (SYC/SQRT_ISWAP) for many layers, optionally ending with a
    measurement layer.

    If `allow_terminal_permutations` is set to `True`, a
    `QuirkQubitPermutationGate` is permitted preceding a measurement.
    Use `circuits2.measure_with_final_permutation` to append a measurement
    gate and track the permutation for implementation via post-processing.
    """
    mom_classes = get_moment_classes(circuit)
    violation_indices = _find_circuit_structure_violations(mom_classes)
    if len(violation_indices) > 0:
        raise BadlyStructuredCircuitError(
            "Badly structured circuit. "
            "Inhomogeneous or non-device moments at indices {}"
                .format(violation_indices))

    # Check that there's only one layer of PhX or Z between entangling layers
    # Check that permutations and measurements must come at the end (in that order)
    # Also count totals while we're at it.
    num_phx = 0
    num_z = 0
    hit_permutation = False
    hit_measurement = False

    tot_n_phx = 0
    tot_n_z = 0
    tot_n_syc = 0
    for mom_class in mom_classes:
        if mom_class in [cirq.PhasedXPowGate, cirq.ZPowGate, cg.SycamoreGate]:
            if hit_permutation:
                raise BadlyStructuredCircuitError(
                    "Permutations must be terminal")
            if hit_measurement:
                raise BadlyStructuredCircuitError(
                    "Measurements must be terminal")

        if mom_class == cirq.PhasedXPowGate:
            num_phx += 1
            tot_n_phx += 1
        elif mom_class == cirq.ZPowGate:
            num_z += 1
            tot_n_z += 1
        elif mom_class == cg.SycamoreGate:
            tot_n_syc += 1
            if num_phx > 1:
                raise BadlyStructuredCircuitError("Too many PhX in this slice")
            if num_z > 1:
                raise BadlyStructuredCircuitError("Too many Z in this slice")
            if num_phx < 1:
                print("Warning: no PhX in this slice")

            num_phx = 0
            num_z = 0
        elif mom_class == cirq.MeasurementGate:
            if hit_measurement:
                raise BadlyStructuredCircuitError("Too many measurements")
            if hit_permutation:
                pass  # fine
            hit_measurement = True
        elif mom_class == cirq.QubitPermutationGate:
            if not allow_terminal_permutations:
                raise BadlyStructuredCircuitError(
                    "Circuit contains permutation gates")

            if hit_measurement:
                raise BadlyStructuredCircuitError("Measurements must be terminal")
            if hit_permutation:
                raise BadlyStructuredCircuitError("Too many permutations")
            hit_permutation = True
        else:
            raise BadlyStructuredCircuitError("Unknown moment class")

    return mom_classes, HomogeneousCircuitStats(tot_n_phx, tot_n_z, tot_n_syc,
                                                hit_permutation,
                                                hit_measurement)


def _idle_qubits_by_moment(circuit: cirq.Circuit):
    qubits = circuit.all_qubits()
    for i, moment in enumerate(circuit.moments):
        idle_qubits = qubits - moment.qubits
        yield i, idle_qubits


def count_circuit_holes(circuit: cirq.Circuit):
    """Count the number of "holes" in a circuit where nothing is happening."""
    return sum(len(iq) for _, iq in _idle_qubits_by_moment(circuit))
