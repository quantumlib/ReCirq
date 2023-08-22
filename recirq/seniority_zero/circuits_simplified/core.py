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

"""Generic circuit routines"""
from typing import List, Optional, Type, Generator

import cirq

from recirq.seniority_zero.circuits_expt.core import starting_pair_loop_on_lattice
from recirq.seniority_zero.circuits_simplified.gates import GSGate


def brickwall_loop_layer(
    qubits: List[cirq.GridQubit], layer_params: List[float], offset: int, gate: Type[cirq.Gate]
) -> Generator:
    """Generates a layer of <gate> acting on a loop of qubits.

    We call gate(layer_params[i])(qubits[(2 * i + offset) % len(qubits)],
                                  qubits[(2 * i + offset + 1) % len(qubits)])
    Args:
        qubits [List[cirq.GridQubit]]: loop of qubits to act on
        layer_params [List[float]]: list of parameters to call.
        offset [int]: how far to shift along qubits
        gate [cirq.Gate]: gate to be applied in brick wall
    """
    pairs = [
        (qubits[(2 * ii + offset) % len(qubits)], qubits[(2 * ii + 1 + offset) % len(qubits)])
        for ii in range(len(qubits))
    ]
    for param, (q0, q1) in zip(layer_params, pairs):
        yield gate(param).on(q0, q1)


def brickwall_givens_swap_network(
    qubits: List[cirq.GridQubit], params: List[float], depth: int, inverse: Optional[bool] = False
) -> Generator:
    """Generates a brickwall Givens swap ansatz with fixed parameters.

    Args:
        qubits [List of cirq.GridQubit]: loop of qubits to act on
        params [List of float]: list of parameters. Parameters are
            ordered first by their layer, then by the position of the
            qubits they act on in the loop
        depth [int]: depth of circuit. We require
            depth * len(qubits) // 2 == len(params)
        inverse [bool]: whether to invert the circuit. As GSGates
            are self-inverse, this can be done by just inverting the order of
            implementing each layer.
    """
    assert depth * len(qubits) // 2 == len(params)
    nparams_layer = len(qubits) // 2
    layer_indices = range(depth)
    if inverse:
        layer_indices = reversed(layer_indices)
    for layer_id in layer_indices:
        layer_params = params[layer_id * nparams_layer : (layer_id + 1) * nparams_layer]
        yield brickwall_loop_layer(qubits, layer_params, layer_id % 2, GSGate)


def GHZ_prep_loop_on_lattice(loop: List[cirq.GridQubit], initial_state: List[int]) -> Generator:
    """Prepares a GHZ state for a loop on a square lattice with NN connectivity.

    Assumes that we start from |000+00...>, where the qubit that begins in |+>
    is given by starting_pair_loop_on_lattice(loop, initial_state)[0].

    Args:
        loop [List of cirq.GridQubit]: qubits to use. Only qubits in loop
            will be used - we effectively loop for the best place to cut the
            loop, create a 4-qubit GHZ state across this cut, then populate
            outward.
        initial_state [List[int]]: A vector b, where we target the preparation
            of |0> + |b>
    """

    init_pair = starting_pair_loop_on_lattice(loop, initial_state)
    yield cirq.CNOT(init_pair[0], init_pair[1])
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
                yield cirq.CNOT(ql, qr)
            else:
                yield cirq.SWAP(ql, qr)
        for qid, (ql, qr) in enumerate(zip(line_piece1[:-1], line_piece1[1:])):
            if sum(state_piece1[qid + 1 :]) == 0:
                continue
            if state_piece1[qid] == 1 or (line_id + qid == 0 and len(line_1) > 2):
                yield cirq.CNOT(ql, qr)
            else:
                yield cirq.SWAP(ql, qr)


def lochschmidt_echo(circuit: cirq.Circuit, qubits: List[cirq.GridQubit]) -> Generator:
    """Makes a lochschmidt echo of a circuit - generates UU^{dag}"""
    circuit = cirq.drop_terminal_measurements(circuit)
    yield circuit
    yield cirq.inverse(circuit)
    yield [cirq.measure(qubit, key=str(qubit)) for qubit in qubits]
