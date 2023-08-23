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

""" Miscellaneous helper functions"""
from typing import Dict, List, Optional

import cirq
import numpy as np
from openfermion import QubitOperator


def get_operator_from_tag(tag: Dict) -> QubitOperator:
    """Turn a EV experiment tag into the operator measured."""
    if tag['type'] == 'Z':
        assert len(tag['ham_ids']) == 1
        qop = QubitOperator('Z{}'.format(*tag['ham_ids']))
    elif tag['type'] == 'ZZ':
        assert len(tag['ham_ids']) == 2
        qop = QubitOperator('Z{} Z{}'.format(*tag['ham_ids']))
    elif tag['type'] == 'XXYYIZZI':
        assert len(tag['ham_ids']) == 2
        qop = (
            tag['sign'] * 0.5 * QubitOperator('X{} X{}'.format(*tag['ham_ids']))
            + tag['sign'] * 0.5 * QubitOperator('Y{} Y{}'.format(*tag['ham_ids']))
            + 0.5 * QubitOperator(f"Z{tag['ham_ids'][0]}")
            + 0.5 * QubitOperator(f"Z{tag['ham_ids'][1]}")
        )
    elif tag['type'] == 'XXYYZZII':
        assert len(tag['ham_ids']) == 2
        qop = (
            tag['sign'] * 0.5 * QubitOperator('X{} X{}'.format(*tag['ham_ids']))
            + tag['sign'] * 0.5 * QubitOperator('Y{} Y{}'.format(*tag['ham_ids']))
            + 0.5 * QubitOperator('Z{} Z{}'.format(*tag['ham_ids']))
            + 0.5 * QubitOperator()
        )
    else:
        raise TypeError('Didnt understand tag')
    return qop


def get_num_circuit_copies(tag: Dict, hamiltonian: QubitOperator, factor: float) -> int:
    """Take a Hamiltonian and a set of circuits, and distribute measurements.

    in the circuit by the weight of the corresponding Hamiltonian term.

    In order to make this distribution work with run_batch, we do not set the
    number of repetitions, but instead we copy each circuit n(circuit) times.
    This function chooses n(circuit).

    n(circuit) overall can be scaled by changing 'factor'; this should be
    chosen such that the number of copies per circuit is close to 1, but such
    that sufficient resolution is achieved.

    Args:
        tag [Dict]: tag for the circuit (needs to be parseable by
            get_operator_from_tag)
        hamiltonian [QubitOperator]: Hamiltonian to use to distribute
            measurements factor [float]: overall scaling of copies
        factor [float]: Scaling factor to multiply everything by to
            balance out the fact that a run_batch call in cirq has the
            same number of repetitions for each circuit.
    """
    operator = get_operator_from_tag(tag)
    term_sum = 0
    for term in operator.terms:
        if term in hamiltonian.terms:
            term_sum += hamiltonian.terms[term]
    return int(np.ceil(np.abs(term_sum) * factor))


def get_standard_qubit_set(
    num_qubits: int, q0: Optional[cirq.GridQubit] = None, direction: Optional[str] = 'R'
) -> List[cirq.GridQubit]:
    '''Create a 'standard' 2xN grid of qubits

    The grid is returned as a loop; e.g. a grid of 8 qubits with right alignment
    takes the form:
        0 - 1 - 2 - 3
        |   |   |   |
        7 - 6 - 5 - 4

    Args:
        num_qubits: How many qubits to put in the loop. If this is odd, it will
            be rounded down.
        q0: The qubit to start from. Defaults to (0,0)
        direction: 'R', 'L', 'U', or 'D'. The direction to go from q0 to
            form the loop. We make a right-hand turn upon reaching
            qubit num_qubits//2 and again the qubit after.
    '''
    if q0 is None:
        q0 = cirq.GridQubit(0, 0)
    if direction == 'R':
        qubits = [q0 + (0, j) for j in range(num_qubits // 2)]
        qubits += [q0 + (1, j) for j in range(num_qubits // 2 - 1, -1, -1)]
    elif direction == 'L':
        qubits = [q0 + (0, -j) for j in range(num_qubits // 2)]
        qubits += [q0 + (-1, -j) for j in range(num_qubits // 2 - 1, -1, -1)]
    elif direction == 'D':
        qubits = [q0 + (j, 0) for j in range(num_qubits // 2)]
        qubits += [q0 + (j, -1) for j in range(num_qubits // 2 - 1, -1, -1)]
    elif direction == 'U':
        qubits = [q0 + (-j, 0) for j in range(num_qubits // 2)]
        qubits += [q0 + (-j, 1) for j in range(num_qubits // 2 - 1, -1, -1)]
    else:
        raise ValueError('I dont understand the given direction')
    return qubits


def get_all_standard_qubit_sets(
    num_qubits: int,
    all_qubits: List[cirq.GridQubit],
) -> List[List[cirq.GridQubit]]:
    """Makes all standard qubit sets of a given size on a fixed lattice"""
    all_ladders = []
    for q0 in all_qubits:
        for direction in ['R', 'L', 'U', "D"]:
            ladder_qubits = get_standard_qubit_set(num_qubits, q0, direction)
            if all([qubit in all_qubits for qubit in ladder_qubits]):
                all_ladders.append(ladder_qubits)
                all_ladders.append(list(reversed(ladder_qubits)))
    return all_ladders


def qubit_hamming_distance(q0: cirq.GridQubit, q1: cirq.GridQubit) -> int:
    return abs(q0.row - q1.row) + abs(q0.col - q1.col)


def check_if_sq_and_preserves_computational_basis(
    gate: cirq.Gate,
    tol: Optional[float]=1e-6
) -> bool:
    """Check if a gate preserves the computational basis up to a phase

    Args:
        gate [cirq.Gate]: the gate to be tested
    """
    # First do some lightweight testing for common gates
    if len(gate.qubits) > 1:
        return False
    elif cirq.protocols.is_measurement(gate):
        return True
    elif isinstance(gate.gate, cirq.PhasedXZGate):
        if (gate.gate.x_exponent + tol) % 1 < 2 * tol:
            return True
        return False
    elif isinstance(gate.gate, cirq.ZPowGate):
        return True
    elif isinstance(gate.gate, (cirq.XPowGate, cirq.YPowGate)):
        if (gate.gate.exponent + tol) % 1 < 2 * tol:
            return True
        return False
    # If we have gotten to here, then I'm not sure what the gate
    # is, so we will have to test the unitary
    unitary = cirq.unitary(gate)
    if abs(unitary[0, 0]) < tol or abs(unitary[0, 1]) < tol:
        return True
    return False


def add_echoes(circuit: cirq.Circuit, check_1q_2q_form: Optional[bool] = True) -> cirq.Circuit:
    """Add echoes to a circuit on all qubits in empty, active spaces

    To not be too smart, we add echoes only when qubits are inactive.

    ***Assumes strict alternating 1q-2q form***

    ***Edits in place***

    The pattern we look for is two or more two-qubit moments that do not
    act on this qubit with empty single-qubit moments in between. We fill
    the area in between with echoes on the single-qubit moments only.
    (To make the net effect of the echo zero, we add an even number of
    echoes to each qubit in each space)

    The 'active space' of a qubit during a circuit is any space between
    the first and last gates that shift it out of the computational basis.

    Args:
        circuit [cirq.Circuit]: the circuit to be altered.
        check_1q_2q_form [bool]: Whether to test whether the circuit is in the
            correct form. Turning this off may increase performance slightly,
            but may produce unexpected results unless you have checked in
            advance.
    """
    if check_1q_2q_form:
        check_separated_single_and_two_qubit_layers([circuit], strict=True)
    for qubit in circuit.all_qubits():

        # Find last moment where this qubit is taken out of the computational
        # basis
        last_moment_id = circuit.prev_moment_operating_on([qubit])
        while (
            check_if_sq_and_preserves_computational_basis(circuit[last_moment_id][qubit]) is True
            and last_moment_id is not None
        ):
            last_moment_id = circuit.prev_moment_operating_on(
                [qubit], end_moment_index=last_moment_id
            )
        if last_moment_id is None:
            break

        # Now find first moment
        first_moment_id = circuit.next_moment_operating_on([qubit])
        while (
            check_if_sq_and_preserves_computational_basis(circuit[first_moment_id][qubit]) is True
            and first_moment_id is not None
        ):
            first_moment_id = circuit.next_moment_operating_on(
                [qubit], start_moment_index=first_moment_id + 1
            )
        if first_moment_id is None:
            raise ValueError('This shouldnt happen.')

        # Now loop through circuit and find empty areas
        current_moment_id = first_moment_id
        while current_moment_id < last_moment_id:
            # Find when the next full moment is
            next_moment_id = circuit.next_moment_operating_on(
                [qubit], start_moment_index=current_moment_id + 1
            )

            # If the next gate is too close, don't insert echoes
            # We need at least 3 free layers starting from a two-qubit layer
            # (Two-qubit layers satisfy current_moment_id % 2 == 1)
            if next_moment_id < current_moment_id + 4 - (current_moment_id % 2):
                current_moment_id = next_moment_id
                continue
            # If we have gotten to this point, we should insert some echoes!
            if current_moment_id % 2 == 0:
                # Shift to a two-qubit layer if not at one.
                current_moment_id += 1
            # We insert one echo for each pair of single and two-qubit layers
            num_echoes = (next_moment_id - current_moment_id + 1) // 4
            for echo_id in range(num_echoes):
                echo_moment_1 = current_moment_id + echo_id * 4 + 1
                echo_moment_2 = current_moment_id + echo_id * 4 + 3
                assert echo_moment_1 < next_moment_id
                circuit[echo_moment_1] += [cirq.X(qubit)]
                try:
                    circuit[echo_moment_2] += [cirq.X(qubit)]
                except ValueError:
                    assert echo_id == num_echoes - 1
                    other_gate = circuit[echo_moment_2][qubit]
                    new_gate = merge_faster(cirq.X(qubit), other_gate)
                    circuit[echo_moment_2] -= other_gate
                    if new_gate is not None:
                        circuit[echo_moment_2] += new_gate
            current_moment_id = next_moment_id
    return circuit


def check_separated_single_and_two_qubit_layers(
    circuits: List[cirq.Circuit], strict: Optional[bool] = True
):
    '''Checks whether a set of circuits satisfy the 1q-2q layer form.

    In the loose 1q-2q layer form, we simply require that each moment in the
    circuit either contains single-qubit or two-qubit gates.

    In the strict 1q-2q layer form, we insist that moments strictly alternate
    (i.e. 1q layer, 2q layer ..., allowing for empty layers.)

    We assume that the last layer contains measurements, and so we ignore it.

    N.B. this does not return True/False, but raises an error

    Args:
        circuits [List[cirq.Circuit]]: the circuits to be checked
        strict [bool]: whether to enforce the strict condition above
    '''
    for circuit_id, circuit in enumerate(circuits):
        for moment_id, moment in enumerate(circuit[:-1]):
            test_set = set([len(gate.qubits) for gate in moment])
            if len(test_set) > 1:
                raise TypeError(f'Circuit: {circuit_id}, moment: {moment_id} is mixed')
            if strict and (
                len(test_set) == 1 and (next(iter(test_set)) % 2 != (moment_id + 1) % 2)
            ):
                raise TypeError(f'Circuit: {circuit_id}, moment: {moment_id} off-position.')


def parallelize_circuits(*circuits: List[cirq.Circuit]) -> cirq.Circuit:
    """Generate single circuit with two circuits acting in parallel."""
    len_new_circuit = max([len(circuit) for circuit in circuits])
    new_circuit = cirq.Circuit()
    for moment_id in range(len_new_circuit):
        new_moment = cirq.Moment()
        for circuit in circuits:
            if moment_id >= len(circuit):
                continue
            new_moment += circuit[moment_id]
        new_circuit.moments.append(new_moment)
    return new_circuit


def safe_tetris_circuits(circuit1: cirq.Circuit, circuit2: cirq.Circuit) -> cirq.Circuit:
    """Tetrises together two circuits, preserving alternating 1q-2q form.

    The term 'tetris' means 'concatenates with shortest depth' - i.e. we try to shift all
    gates in circuit2 as far as possible to the left."""
    if len(circuit2) == 0:
        return circuit1[:]
    if len(circuit1) == 0:
        return circuit2[:]

    moment_to_tetris_to = circuit1.prev_moment_operating_on(circuit2.all_qubits())
    if moment_to_tetris_to is None:  # No overlap
        circuit = parallelize_circuits(circuit1, circuit2)
        return circuit

    moment_1r = circuit1[moment_to_tetris_to]
    moment_2l = circuit2[0]
    test_set_1r = set([len(gate.qubits) for gate in moment_1r])
    test_set_2l = set([len(gate.qubits) for gate in moment_2l])
    if len(test_set_1r) != 1:
        raise ValueError('Circuit 1 not in correct form, moments {}, {}'.format(
            moment_1r, moment_2l))
    if len(test_set_2l) != 1:
        raise ValueError('Circuit 2 not in correct form, moments {}, {}'.format(
            moment_1r, moment_2l))
    (gate_type_1r,) = test_set_1r
    (gate_type_2l,) = test_set_2l

    if gate_type_1r == 1:
        if gate_type_2l == 1:
            circuit = safe_concatenate_circuits(circuit1[: moment_to_tetris_to + 1], circuit2[0:1])
            circuit = safe_concatenate_circuits(
                circuit, parallelize_circuits(circuit1[moment_to_tetris_to + 1 :], circuit2[1:])
            )
        elif gate_type_2l == 2:
            circuit = safe_concatenate_circuits(
                circuit1[: moment_to_tetris_to + 1],
                parallelize_circuits([circuit2, circuit1[moment_to_tetris_to + 1 :]]),
            )
        else:
            raise ValueError(f'Dont understand gate_type_2l={gate_type_2l}')
    elif gate_type_1r == 2:
        moment_1l = circuit1[moment_to_tetris_to + 1]
        test_set_1l = set([len(gate.qubits) for gate in moment_1l])
        if len(test_set_1l) != 1:
            raise ValueError('Circuit 1 not in correct form, moments {}, {}, {}'.format(
                moment_1r, moment_1l, moment_2l))
        (gate_type_1l,) = test_set_1l
        if gate_type_1l == gate_type_2l:
            circuit = safe_concatenate_circuits(
                circuit1[: moment_to_tetris_to + 1],
                parallelize_circuits(circuit1[moment_to_tetris_to + 1 :], circuit2),
            )
        else:
            circuit = safe_concatenate_circuits(
                circuit1[: moment_to_tetris_to + 2],
                parallelize_circuits(circuit1[moment_to_tetris_to + 2 :], circuit2),
            )
    else:
        raise ValueError(f'Dont understand gate_type_1r={gate_type_1r}')
    return circuit


def safe_concatenate_circuits(circuit1: cirq.Circuit, circuit2: cirq.Circuit) -> cirq.Circuit:
    """Concatenates two circuits while preserving 1q-2q form."""
    if len(circuit2) == 0:
        return circuit1[:]
    if len(circuit1) == 0:
        return circuit2[:]
    moment_1r = circuit1[-1]
    moment_2l = circuit2[0]
    test_set_1r = set([len(gate.qubits) for gate in moment_1r])
    test_set_2l = set([len(gate.qubits) for gate in moment_2l])

    if len(test_set_1r) != 1:
        raise ValueError('Circuit 1 not in correct form, moments {}, {}'.format(
            moment_1r, moment_2l))
    if len(test_set_2l) != 1:
        raise ValueError('Circuit 2 not in correct form, moments {}, {}'.format(
            moment_1r, moment_2l))
    (gate_type_1r,) = test_set_1r
    (gate_type_2l,) = test_set_2l

    if gate_type_1r == 1 and gate_type_2l == 1:
        new_middle_moment = cirq.Moment()
        for gate_1r in moment_1r:
            if gate_1r.qubits[0] in moment_2l.qubits:
                gate_2l = moment_2l[gate_1r.qubits[0]]
                new_gate = merge_faster(gate_1r, gate_2l)
                if new_gate is not None:
                    new_middle_moment += [new_gate]
            else:
                new_middle_moment += [gate_1r]
        for gate_2l in moment_2l:
            if gate_2l.qubits[0] not in moment_1r.qubits:
                new_middle_moment += [gate_2l]

        circuit = cirq.Circuit()
        circuit._moments = circuit1.moments[:-1]
        circuit._moments.append(new_middle_moment)
        circuit._moments += circuit2.moments[1:]
    else:
        circuit = cirq.Circuit()
        circuit._moments = circuit1.moments
        circuit._moments += circuit2.moments
    return circuit


def merge_faster(gate1: cirq.Gate, gate2: cirq.Gate, tol: Optional[float]=1e-6) -> cirq.Gate:
    """Merges single qubit gates to a phasedXZ gate, is faster than cirq when gates are rz."""
    if (
        type(gate1.gate) == cirq.PhasedXZGate
        and type(gate2.gate) == cirq.PhasedXZGate
        and abs(gate2.gate.x_exponent) < tol
    ):
        new_gate = cirq.PhasedXZGate(
            axis_phase_exponent=gate1.gate.axis_phase_exponent,
            x_exponent=gate1.gate.x_exponent,
            z_exponent=gate1.gate.z_exponent + gate2.gate.z_exponent,
        )
    else:
        new_unitary = cirq.unitary(gate2.gate) @ cirq.unitary(gate1.gate)
        new_gate = cirq.single_qubit_matrix_to_phxz(new_unitary)
        if new_gate is None:
            return None
    return new_gate.on(gate1.qubits[0])
