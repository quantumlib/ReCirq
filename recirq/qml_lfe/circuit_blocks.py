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

"""Sycamore native circuit building blocks."""

import os
import cirq
import cirq_google
import functools


@functools.lru_cache
def _load_circuit(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "r") as f:
        return cirq.read_json(f)


def tsym_block(qubits, params):
    """Create a tsym block.

    Y axis rotations followed by a real entried permutation operation.

    Args:
        qubits: The pair of qubits to apply this operation on.
        params: The length 2 list of Y rotation angles.

    Returns:
        A list of `cirq.Operations` realizing a tsym block.
    """
    mapped_circuit = _load_circuit("tsym_permute.txt").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    rots = [cirq.Y(qubits[0]) ** params[0], cirq.Y(qubits[1]) ** params[1]]
    return rots + list(mapped_circuit.all_operations())


def _get_op(code):
    if code <= 4 / 3:
        return cirq.X ** 0.5
    elif code <= 2 * (4 / 3):
        return cirq.Y ** 0.5
    elif code <= 3 * (4 / 3):
        return cirq.PhasedXZGate(axis_phase_exponent=0.25, x_exponent=0.5, z_exponent=0)

    raise ValueError("panic!")


def scrambling_block(qubits, params):
    """Create a scrambling block.

    Randomly applies one of sqrt(X), sqrt(Y) and sqrt(W) on qubits followed
    by a SycamoreGate.

    Args:
        qubits: The pair of qubits to create a scrambling block on.
        params: List of two random numbers between 0 and 4 to decide on
            single qubit operations for each qubit.

    Returns:
        A list of `cirq.Operations` realizing a scrambling block.
    """
    a_op = _get_op(params[0])
    b_op = _get_op(params[1])
    return [a_op(qubits[0]), b_op(qubits[1]), cirq_google.SycamoreGate()(*qubits)]


def block_1d_circuit(qubits, depth, block_fn, random_source):
    """Create a 1D block structure circuit using block_fn.

    Alternate couplings layer by layer constructing a circuit that
    looks like:

    a --- b XXX c --- d XXX e --- f
    a XXX b --- c XXX d --- e XXX f
    a --- b XXX c --- d XXX e --- f
    a XXX b --- c XXX d --- e XXX f


    Where each XXX is constructed by calling `block_fn` on each pair of qubits,
    using the next unused entry in `random_source`.

    Args:
        qubits: List of adjacent qubits.
        depth: Number of layers to include in circuit.
        block_fn: Python function that accepts qubit pairs and
            parameters to generate interactions on those qubit pairs.
        random_source: Python list or np.ndarray with shape:
            [depth * len(qubits) // 2, n_params_for_block_fn].

    """
    ops = []
    i_random = 0
    for t in range(depth):
        for i in range(0 if t % 2 == 0 else 1, len(qubits) - 1, 2):
            ops += block_fn([qubits[i], qubits[i + 1]], random_source[i_random])
            i_random += 1

    return cirq.Circuit(ops)


def bell_pair_block(qubits):
    """Creates a bell pair on qubits.

    Enacts H(a) + CNOT(a, b) using SycamoreGates and single qubit operations.

    Args:
        qubits: The qubits to prepare the bell pair on.

    Returns:
         A list of `cirq.Operations` realizing a bell pair.
    """
    mapped_circuit = _load_circuit("syc_bell_pair.txt").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    return mapped_circuit.all_operations()


def un_bell_pair_block(qubits):
    """Un-bell-pair on qubits.

    Enacts CNOT(a, b) + H(a) using SycamoreGates and single qubit operations.

    Args:
        qubits: The qubits to un-prepare the bell pair on.

    Returns:
         A list of `cirq.Operations` realizing the operation.
    """
    mapped_circuit = _load_circuit("syc_un_bell_pair.txt").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    return mapped_circuit.all_operations()


def swap_block(qubits):
    """Swap two qubits.

    Enacts SWAP(a, b)using SycamoreGates and single qubit operations.

    Args:
        qubits: The qubits to swap.

    Returns:
         A list of `cirq.Operations` realizing the swap.
    """
    mapped_circuit = _load_circuit("syc_swap.txt").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    return mapped_circuit.all_operations()
