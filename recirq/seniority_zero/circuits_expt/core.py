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

"""Implementation of the actual circuits used in the s0 experiment

To ensure precise scheduling + echo insertion, much of this is hard-coded and
long-winded. We recommend using circuits_simplified to understand what is going
on in the experiment.
"""
from typing import List, Optional

import cirq

from recirq.seniority_zero.circuits_expt.gates_this_experiment import cmnot, cnot, gsgate, swap
from recirq.seniority_zero.misc import (
    parallelize_circuits,
    safe_concatenate_circuits,
    safe_tetris_circuits,
)
from recirq.seniority_zero.scheduling import get_tqbg_groups


def _make_lightcone(num_qubits, depth=None):
    """Make list of gates which are hit by a given qubit's backwards light cone.
    Assumes a brickwall ansatz
    """
    if depth is None:
        depth = num_qubits // 2
    gates_per_layer = num_qubits // 2
    last_layer_offset = (gates_per_layer + 1) % 2
    timelike_gates_all_qubits = []
    for qid in range(num_qubits):
        # Add gate that this qubit interacts with in the last layer
        timelike_gates_this_qubit = [[((qid - last_layer_offset) % num_qubits) // 2]]
        # Populate backwards
        for layer_id in range(1, num_qubits // 2):
            # All gate ids from the previous layer are included here
            timelike_gates_this_qubit.append(list(timelike_gates_this_qubit[-1]))

            # Either add the largest gid +1 or the smallest gid -1
            offset = (layer_id + last_layer_offset) % 2
            if offset == 0:
                timelike_gates_this_qubit[-1].append(
                    (timelike_gates_this_qubit[-1][-1] + 1) % gates_per_layer
                )
            else:
                timelike_gates_this_qubit[-1].insert(
                    0, (timelike_gates_this_qubit[-1][0] - 1) % gates_per_layer
                )
        timelike_gates_all_qubits.append(timelike_gates_this_qubit)
    return timelike_gates_all_qubits


# Indexed: _timelike_gates[num_qubits][qid][negative_layer_id]
_timelike_gates = {num_qubits: _make_lightcone(num_qubits) for num_qubits in [4, 6, 8, 10, 12]}


def gs_layer(qubits: List, layer_params: List[float], offset: int, gate_indices: List[int]):
    """Single even layer of the brick wall givens-swap circuit.

    Qubits are coupled as (dashed lines represent a single givens-swap gate)
    0-1 2-3 4-5 6-7 8-9
    or (if odd_flag == True)
    0 1-2 3-4 5-6 7-8 9
    |_________________|

    Args:
        qubits [List[cirq.Qid]]: array with the indexing for the qubit
        layer_params [List[float]]: array of length len(qubits)//2 with
            the parameters for the givens rotations of the layer
        odd_flag [bool]: whether this is an even or odd layer
    """
    assert len(layer_params) == len(qubits) // 2

    circuits = [
        gsgate(qubits[2 * i + offset], qubits[(2 * i + 1 + offset) % len(qubits)], layer_params[i])
        for i in gate_indices
    ]
    circuit = parallelize_circuits(*circuits)

    return circuit


def givens_swap_network(
    qubits: List,
    params: List[float],
    depth: int,
    target_qids: Optional[List[cirq.Qid]] = None,
    inverse: Optional[bool] = False,
    global_echo: Optional[bool] = False,
    echoed_layers: Optional[List] = None,
):
    """Create a brick wall givens-swap circuit.

    Args:
        qubits [list]: array with the indexing for the qubit
        params [list]: array of length depth*(len(qubits)//2)
            with the parameters for the givens rotations
        depth [int]: depth of the BrickLayer network
        inverse [bool]: whether to invert the circuit (the GS gate is
            self-inverse, so this just requires flipping the order of
            application)
        global_echo [bool]: whether the circuit has incurred a global
            echo (i.e. prod_i X_i before and afterwards)
        echoed_layers [List]: which layers to put additional echos **before**.
    """

    num_qubits = len(qubits)
    nparams_layer = num_qubits // 2
    echo_flag = global_echo
    layer_indices = range(depth)
    if inverse:
        layer_indices = reversed(layer_indices)
    if echoed_layers is None:
        echoed_layers = []

    echo_layer = cirq.Circuit([cirq.X(qubit) for qubit in qubits])
    circuit = cirq.Circuit()
    for layer_id in layer_indices:
        if layer_id in echoed_layers:
            circuit = safe_concatenate_circuits(circuit, echo_layer)
            echo_flag = not echo_flag
        offset = layer_id % 2
        if target_qids:
            gate_indices = set(
                [
                    gid
                    for qid in target_qids
                    for gid in _timelike_gates[num_qubits][qid][depth - layer_id - 1]
                ]
            )
        else:
            gate_indices = range(len(qubits) // 2)
        layer_params = params[layer_id * nparams_layer : (layer_id + 1) * nparams_layer]
        if echo_flag is True:
            layer_params = [-xx for xx in layer_params]
        layer_circuit = gs_layer(
            qubits=qubits, layer_params=layer_params, offset=layer_id % 2, gate_indices=gate_indices
        )
        circuit = safe_concatenate_circuits(circuit, layer_circuit)
    if len(echoed_layers) % 2 == 1:
        circuit = safe_concatenate_circuits(circuit, echo_layer)
    return circuit


def GHZ_prep_2xn_mixed_filling(
    qubit_line1: List,
    qubit_line2: List,
    state_line1: List[int],
    state_line2: List[int],
    starting_index: int,
    inverse: Optional[bool] = False,
    global_echo: Optional[bool] = False,
):
    """Prepare a GHZ state on a 2xn qubit ladder

    between a computational basis state and the vacuum state,
    starting from an initial qubit prepared in |0>+|1>.

    Assumes qubit_line1[j] and qubit_line2[j] are coupled,
    and linear coupling along both lines.

    Args:
        qubit_line1 [list]: list of qubits on line 1
        qubit_line2 [list]: list of qubits on line 2
        state_line1 [list]: computational basis state on line 1
        state_line2 [list]: computational basis state on line 2
        starting_index [int]: where to begin filling from.
    """

    if global_echo is True:
        cnot_or_cmnot = cmnot
    else:
        cnot_or_cmnot = cnot

    filled_indices = [
        index
        for index in range(len(qubit_line1))
        if (state_line1[index] == 1 or state_line2[index] == 1)
    ]

    # First spread out to the right
    right_gates = []
    for qr_index in range(starting_index + 1, max(filled_indices) + 1):
        if qr_index - 1 == starting_index or qr_index - 1 in filled_indices:
            right_gates.append(cnot_or_cmnot(qubit_line1[qr_index - 1], qubit_line1[qr_index]))
        else:
            right_gates.append(swap(qubit_line1[qr_index - 1], qubit_line1[qr_index]))

    # Then spread out to the left
    left_gates = []
    for ql_index in range(starting_index - 1, min(filled_indices) - 1, -1):
        if ql_index + 1 in filled_indices:
            left_gates.append(cnot_or_cmnot(qubit_line1[ql_index + 1], qubit_line1[ql_index]))
        else:
            left_gates.append(swap(qubit_line1[ql_index + 1], qubit_line1[ql_index]))

    # Now combine layers
    combined_gates = [right_gates[0]]
    for left_gate, right_gate in zip(left_gates, right_gates[1:]):
        combined_gates.append(parallelize_circuits(right_gate, left_gate))

    # Add the last layers here if the circuit is asymmetric
    # At the same time, we test if the last layer of gates can be merged
    # with this layer.
    if len(left_gates) > len(right_gates) - 1:
        combined_gates += left_gates[len(right_gates) - 1 :]
        if state_line2[min(filled_indices)] == 1 or state_line2[min(filled_indices) + 1] == 1:
            combined_gates.append(cirq.Circuit())
    elif len(right_gates) > len(left_gates) + 1:
        combined_gates += right_gates[len(left_gates) + 1 :]
        if state_line2[max(filled_indices)] == 1 or state_line2[max(filled_indices) - 1] == 1:
            combined_gates.append(cirq.Circuit())
    else:
        if (
            state_line2[max(filled_indices)] == 1
            or state_line2[max(filled_indices) - 1] == 1
            or state_line2[min(filled_indices)] == 1
            or state_line2[min(filled_indices) + 1] == 1
        ):
            combined_gates.append(cirq.Circuit())

    # Now spread out in the second dimension
    last_layer_gates = []
    for index in filled_indices:
        if state_line2[index] == 1:
            if state_line1[index] == 1:
                last_layer_gates.append(cnot_or_cmnot(qubit_line1[index], qubit_line2[index]))
            else:
                last_layer_gates.append(swap(qubit_line1[index], qubit_line2[index]))

    combined_gates[-1] = parallelize_circuits(combined_gates[-1], *last_layer_gates)

    # All of the individual layers are self-inverse, so we can just
    # invert the order of the gates to inverse the circuit.
    if inverse:
        combined_gates = reversed(combined_gates)

    # Now concatentate everything
    circuit = cirq.Circuit()
    for subcircuit in combined_gates:
        circuit = safe_concatenate_circuits(circuit, subcircuit)
    return circuit


def starting_index_GHZ_prep_2xn(state_line1, state_line2):
    """Determines the optimal index for GHZ preparation

    Finds the index on a 2xn grid that is optimally positioned
    for GHZ preparation (in terms of the resulting circuit depth)

    Args:
        state_line1 [list]: computational basis state on line 1
        state_line2 [list]: computational basis state on line 2
    """

    filled_indices = [
        index
        for index in range(len(state_line1))
        if (state_line1[index] == 1 or state_line2[index] == 1)
    ]
    starting_index = (max(filled_indices) + min(filled_indices)) // 2
    return starting_index


def get_starting_qubits_all_groups_2xN(qubits, extra_shift=None):
    """Get (group-dependent) list of which qubit contains the VPE information
    (This uses the depreciated GHZ preparation methods above)
    """
    print(f'Using extra shift: {extra_shift}')
    num_qubits = len(qubits)
    groups = get_tqbg_groups(qubits)
    starting_qubits = []
    for group in groups:
        # Data for GHZ preparation
        if extra_shift is None:
            extra_shift = (num_qubits % 4) // 2
        shift = group.shift
        # This state matches the ordering of qubits
        starting_state = [(qid + shift + extra_shift) % 2 for qid in range(num_qubits)]

        # Need to reverse the ordering of the second row of qubits
        # to make this consistent with the ordering for the GHZ prep routine
        state_line1 = starting_state[: num_qubits // 2]
        state_line2 = list(reversed(starting_state[num_qubits // 2 :]))
        qubit_line1 = qubits[: num_qubits // 2]
        qubit_line2 = list(reversed(qubits[num_qubits // 2 :]))
        q0id = starting_index_GHZ_prep_2xn(state_line1, state_line2)
        q0 = qubit_line1[q0id]
        starting_qubits.append(q0)
    return starting_qubits


def get_starting_qubits_all_groups_loop(qubits, groups, extra_shift=None):
    """Get (group-dependent) list of which qubit contains the VPE information"""
    num_qubits = len(qubits)
    starting_qubits = []
    for group in groups:
        # Data for GHZ preparation
        if extra_shift is None:
            extra_shift = (num_qubits % 4) // 2
        shift = group.shift
        # This state matches the ordering of qubits
        initial_state = [(qid + shift + extra_shift) % 2 for qid in range(num_qubits)]

        # Need to reverse the ordering of the second row of qubits
        # to make this consistent with the ordering for the GHZ prep routine
        q0id = starting_index_loop_on_lattice(qubits, initial_state)
        q0 = qubits[q0id]
        starting_qubits.append(q0)
    return starting_qubits


def starting_index_loop_on_lattice(loop: List[cirq.GridQubit], initial_state: List[int]):
    """Finds the starting index of an optimal circuit to
    produce a GHZ state on a lattice.
    """
    pair = starting_pair_loop_on_lattice(loop, initial_state)
    q0id = loop.index(pair[0])
    return q0id


def starting_pair_loop_on_lattice(loop: List[cirq.GridQubit], initial_state: List[int]):
    """Finds the starting pair of an optimal circuit to
    produce a GHZ state on a lattice.

    Assumes that a Bell state will first be prepared on these
    two qubits, then copied to the rest of the system around
    the loop."""
    adjacent_pairs = [
        (q0, q1) for q0id, q0 in enumerate(loop) for q1 in loop[q0id + 1 :] if q0.is_adjacent(q1)
    ]

    best_pair = None
    best_depth = None
    for pair in adjacent_pairs:
        q0id = loop.index(pair[0])
        q1id = loop.index(pair[1])
        depth_inner = (q1id - q0id - 1) // 2  # N.B. should be an integer anyway!
        depth_outer = (len(loop) - (q1id - q0id) - 1) // 2  # N.B this also.
        depth = max(depth_inner, depth_outer)
        if depth_inner == depth_outer:
            depth += 1
        if best_depth is None or depth < best_depth:
            best_pair = pair
            best_depth = depth
    return best_pair


def GHZ_prep_loop_on_lattice(
    loop: List[cirq.GridQubit],
    initial_state: List[int],
    global_echo: Optional[bool] = False,
    inverse: Optional[bool] = False,
):

    if global_echo is True:
        cnot_or_cmnot = cmnot
    else:
        cnot_or_cmnot = cnot

    init_pair = starting_pair_loop_on_lattice(loop, initial_state)
    circuit_list = [cnot_or_cmnot(init_pair[0], init_pair[1])]
    q0id = loop.index(init_pair[0])
    q1id = loop.index(init_pair[1])
    inner_line = loop[q0id : q1id + 1]
    inner_state = initial_state[q0id : q1id + 1]
    outer_line = list(reversed(loop[q1id:] + loop[: q0id + 1]))
    outer_state = list(reversed(initial_state[q1id:] + initial_state[: q0id + 1]))
    if len(inner_line) <= len(outer_line):
        line_0 = outer_line
        state_0 = outer_state
        line_1 = inner_line
        state_1 = inner_state
    else:
        line_0 = inner_line
        state_0 = inner_state
        line_1 = outer_line
        state_1 = outer_state

    for line_id, line, state in [[0, line_0, state_0], [1, line_1, state_1]]:
        depth = len(line) // 2

        # Cut line into bit near q0 and bit near q1
        line_piece0 = line[:depth]
        state_piece0 = state[:depth]
        line_piece1 = list(reversed(line[depth:]))
        state_piece1 = list(reversed(state[depth:]))

        # Run over each piece, swapping or cnotting till we get to the end
        for qid, (ql, qr) in enumerate(zip(line_piece0[:-1], line_piece0[1:])):
            if sum(state_piece0[qid + 1 :]) == 0:
                continue
            if state_piece0[qid] == 1 or (line_id + qid == 0 and len(line_1) > 2):
                circuit = cnot_or_cmnot(ql, qr)
            else:
                circuit = swap(ql, qr)
            circuit_list.append(circuit)
        for qid, (ql, qr) in enumerate(zip(line_piece1[:-1], line_piece1[1:])):
            if sum(state_piece1[qid + 1 :]) == 0:
                continue
            if state_piece1[qid] == 1 or (line_id + qid == 0 and len(line_1) > 2):
                circuit = cnot_or_cmnot(ql, qr)
            else:
                circuit = swap(ql, qr)
            circuit_list.append(circuit)

    # All of our gates are self inverse, so we can flip the order
    # here to invert the circuit. Note that this works only on the
    # cnot / swap blocks, and not their sub-units.
    if inverse is True:
        circuit_list = reversed(circuit_list)

    # Compile everything!
    circuit = cirq.Circuit()
    for new_circuit in circuit_list:
        circuit = safe_tetris_circuits(circuit, new_circuit)

    return circuit
