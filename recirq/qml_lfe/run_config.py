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

"""Runtime configuration items for excuting experiments on Weber (or sim)."""

from typing import Dict, List
import os
import cirq
import cirq_google


def flatten_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Pack operations in circuit to the left as far as possible.

    Args:
        circuit: `cirq.Circuit` who's operations will be packed.

    Returns:
        A `cirq.Circuit` with operations packed to the left as
        far as possible.
    """
    return cirq.Circuit([op for mom in circuit for op in mom])


def execute_batch(
    batch: List[cirq.Circuit], n_shots: int, use_engine: bool
) -> List[cirq.Result]:
    """Use either simulator or engine to execute a batch of circuits.

    Args:
        batch: List of `cirq.Circuit`s to execute.
        n_shots: Number of shots to draw from all circuits.
        use_engine: Whether or not to make use of quantum engine and the
            weber processor. Note this requires the GOOGLE_CLOUD_PROJECT
            environment variable to be set, along with the required cloud
            permissions.
    Returns:
        List of results to execution.
    """
    runner = None
    if use_engine:
        runner = cirq_google.get_engine_sampler(
            processor_id="weber", gate_set_name="sycamore"
        )
    else:
        runner = cirq.Simulator()

    res = runner.run_batch(
        programs=batch,
        repetitions=n_shots,
    )
    return [x[0] for x in res]


def execute_sweep(
    circuit: cirq.Circuit, params: List[Dict[str, float]], use_engine: bool
) -> List[cirq.Result]:
    """Use either simulator or engine to run a one rep sweep on a circuit.

    Args:
        circuit: the `cirq.Circuit` with symbols to execute.
        params: The list of parameters to place in the symbols.
        use_engine: Whether or not to make use of quantum engine and the
            weber processor. Note this requires the GOOGLE_CLOUD_PROJECT
            environment variable to be set, along with the required cloud
            permissions.
    Returns:
        List of results to execution.
    """
    runner = None
    if use_engine:
        runner = cirq_google.get_engine_sampler(
            processor_id="weber", gate_set_name="sycamore"
        )
    else:
        runner = cirq.Simulator()

    return runner.run_sweep(program=circuit, params=params, repetitions=1)


def qubit_pairs() -> List[List[cirq.Qid]]:
    """Returns a pair of valid system qubits for 1D tsym vs scrambling.

    Returns pairs of qubits forming a valid 1D chain on the
    weber processor.

    Note: in order for a 1D chain to be valid, x[i][0] must be
        adjacent to x[i + 1][0] and x[i][0] must be adjacent to x[i][1]
        for all x in the returned system pairs.

    Returns:
        A list of valid system pairs.
    """
    return [
        [cirq.GridQubit(1, 5), cirq.GridQubit(0, 5)],
        [cirq.GridQubit(1, 6), cirq.GridQubit(0, 6)],
        [cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)],
        [cirq.GridQubit(2, 7), cirq.GridQubit(1, 7)],
        [cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)],
        [cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)],
        [cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)],
        [cirq.GridQubit(5, 6), cirq.GridQubit(4, 6)],
        # Avoid dead qubit 5, 4.
        # [cirq.GridQubit(5, 5), cirq.GridQubit(5, 4)],
        [cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)],
        [cirq.GridQubit(6, 5), cirq.GridQubit(5, 5)],
        [cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)],
        [cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)],
        [cirq.GridQubit(7, 3), cirq.GridQubit(7, 2)],
        [cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)],
        [cirq.GridQubit(6, 2), cirq.GridQubit(6, 1)],
        [cirq.GridQubit(5, 2), cirq.GridQubit(5, 1)],
        [cirq.GridQubit(4, 2), cirq.GridQubit(4, 1)],
        [cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)],
        [cirq.GridQubit(3, 3), cirq.GridQubit(3, 2)],
        [cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)],
        [cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)],
        [cirq.GridQubit(2, 5), cirq.GridQubit(2, 4)],
    ]
