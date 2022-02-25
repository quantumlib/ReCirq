# Copyright 2021 Google
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
from typing import Callable, Dict, List, Tuple
from numbers import Number

import os
import cirq
import cirq_google
import functools
import numpy as np


@functools.lru_cache(maxsize=128)
def _load_circuit(fname: str) -> cirq.Circuit:
    with open(os.path.join(os.path.dirname(__file__), fname), "r") as f:
        return cirq.read_json(f)


def tsym_block(qubits: List[cirq.Qid], params: List[Number]) -> List[cirq.Operation]:
    """Create a tsym block.

    Y axis rotations followed by a real entried permutation operation.

    Args:
        qubits: The pair of qubits to apply this operation on.
        params: The length 2 list of Y rotation angles.

    Returns:
        A list of `cirq.Operations` realizing a tsym block.
    """
    mapped_circuit = _load_circuit("tsym_permute.json").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    rots = [cirq.Y(qubits[0]) ** params[0], cirq.Y(qubits[1]) ** params[1]]
    return rots + list(mapped_circuit.all_operations())


def _get_op(code: Number) -> cirq.Gate:
    if code <= 4 / 3:
        return cirq.X ** 0.5
    elif code <= 2 * (4 / 3):
        return cirq.Y ** 0.5
    elif code <= 3 * (4 / 3):
        return cirq.PhasedXZGate(axis_phase_exponent=0.25, x_exponent=0.5, z_exponent=0)

    raise ValueError("Random codes must be between 0 and 4.")


def scrambling_block(
    qubits: List[cirq.Qid], params: List[Number]
) -> List[cirq.Operation]:
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


def block_1d_circuit(
    qubits: List[cirq.Qid],
    depth: int,
    block_fn: Callable,
    random_source: List[List[Number]],
) -> cirq.Circuit:
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

    Returns:
        A 1d block circuit.
    """
    ops = []
    i_random = 0
    for t in range(depth):
        for i in range(0 if t % 2 == 0 else 1, len(qubits) - 1, 2):
            ops += block_fn([qubits[i], qubits[i + 1]], random_source[i_random])
            i_random += 1

    return cirq.Circuit(ops)


def bell_pair_block(qubits: List[cirq.Qid]) -> List[cirq.Operation]:
    """Creates a bell pair on two qubits.

    Enacts H(a) + CNOT(a, b) using SycamoreGates and single qubit operations.

    Args:
        qubits: The qubits to prepare the bell pair on.

    Returns:
         A list of `cirq.Operations` realizing a bell pair.
    """
    mapped_circuit = _load_circuit("syc_bell_pair.json").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    return mapped_circuit.all_operations()


def un_bell_pair_block(qubits: List[cirq.Qid]) -> List[cirq.Operation]:
    """Un compute a bell pair on two qubits.

    Enacts CNOT(a, b) + H(a) using SycamoreGates and single qubit operations.

    Args:
        qubits: The qubits to un-prepare the bell pair on.

    Returns:
         A list of `cirq.Operations` realizing the operation.
    """
    mapped_circuit = _load_circuit("syc_un_bell_pair.json").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    return mapped_circuit.all_operations()


def swap_block(qubits: List[cirq.Qid]) -> List[cirq.Operation]:
    """Swap two qubits.

    Enacts SWAP(a, b) using SycamoreGates and single qubit operations.

    Args:
        qubits: The qubits to swap.

    Returns:
         A list of `cirq.Operations` realizing the swap.
    """
    mapped_circuit = _load_circuit("syc_swap.json").transform_qubits(
        {cirq.GridQubit(0, 0): qubits[0], cirq.GridQubit(0, 1): qubits[1]}
    )
    return mapped_circuit.all_operations()


def inv_z_basis_gate(pauli: str) -> cirq.Gate:
    """Returns inverse Z basis transformation ops for a given Pauli.

    Args:
        pauli: Python str representing a single pauli.

    Returns:
        Corresponding `cirq.Gate` to do the inverse basis conversion.
    """
    if pauli == "I" or pauli == "Z":
        return cirq.I
    if pauli == "X":
        return cirq.H
    if pauli == "Y":
        # S^dag H to get to computational, H S to go back.
        return cirq.PhasedXZGate(
            axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-0.5
        )
    raise ValueError("Invalid Pauli.")


def create_randomized_sweeps(
    hidden_p: np.ndarray,
    symbols: Tuple[str],
    n_params: int,
    rand_state: np.random.RandomState,
) -> List[Dict[str, int]]:
    """Generate sweeps to help prepare \rho = 2^(-n) (I + \alpha P) states.

    See section A.2.a (https://arxiv.org/pdf/2112.00778.pdf) more details.

    Args:
        hidden_p: Pauli operator in (I + \alpha P)
        symbols: Symbols to generate values for to prepare \rho.
        n_params: Number of unique parameter configurations (sweeps)
            to generate for the given `symbols`.
        rand_state: np.random.RandomState source of randomness.

    Returns:
        List of sweeps, that when placed in a circuit will realize
        \rho.
    """

    last_i = 0
    for i, pauli in enumerate(hidden_p):
        if pauli != "I":
            last_i = i

    sign_p = rand_state.choice([1, -1])
    all_sweeps = []
    for _ in range(n_params):
        current_sweep = dict()
        for twocopy in [0, 1]:
            parity = sign_p * rand_state.choice([1, -1], p=[0.95, 0.05])
            for i, pauli in enumerate(hidden_p):
                current_symbol = symbols[2 * i + twocopy]
                current_sweep[current_symbol] = rand_state.choice([0, 1])
                if pauli != "I":
                    if last_i == i:
                        v = 1 if parity == -1 else 0
                        current_sweep[current_symbol] = v
                    elif current_sweep[current_symbol] == 1:
                        parity *= -1

        all_sweeps.append(current_sweep)

    return all_sweeps
