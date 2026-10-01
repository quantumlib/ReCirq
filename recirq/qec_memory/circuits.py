# Copyright 2026 Google
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

import cirq
import numpy as np


def get_pre_loop_key(keys: list[str]) -> int:
    """Find the last measure key before the ones from within the CircuitOperation.

    Args:
        keys: A list of the measure keys.

    Returns:
        The key as an integer.

    Raises:
        ValueError: If keys does not contain some non-consecutive integers strings.
    """
    digit_keys = sorted([int(key) for key in keys if key.isdigit()])
    for i in range(len(digit_keys) - 1):
        if digit_keys[i + 1] - digit_keys[i] > 1:
            return digit_keys[i]
    raise ValueError(
        "`keys` must contain some integer strings and they must not all be consecutive"
    )


def replace_loop_repetitions(
    circuit: cirq.Circuit, cycles: int, original_cycles: int = 10
) -> cirq.Circuit:
    """Replace the number of times a CircuitOperation is repeated within a circuit.

    Args:
        circuit: A circuit containing a single CircuitOperation.
        cycles: The desired number of QEC cycles.
        original_cycles: The number of QEC cycles in the input circuit.

    Returns:
        A modified circuit.
    """
    new_moments = []
    for moment in circuit:
        if len(moment.operations) == 1 and isinstance(
            moment.operations[0], cirq.CircuitOperation
        ):
            op = moment.operations[0]
            new_moments.append(
                cirq.Moment(
                    op.replace(
                        repetitions=cycles - original_cycles + op.repetitions,
                        repetition_ids=None,
                    )
                )
            )
        else:
            new_moments.append(moment)
    return cirq.Circuit.from_moments(*new_moments)


def identify_measure_qubits(circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    """Identify which qubits are used for mid-circuit measurements.

    Args:
        circuit: The circuit to inspect.

    Returns:
        The measure qubits.
    """
    measure_qubits = set()
    for op in circuit[:-1].all_operations():
        if cirq.is_measurement(op) and not isinstance(op, cirq.CircuitOperation):
            for q in op.qubits:
                measure_qubits.add(q)
    return measure_qubits


def identify_terminal_measured_qubits(circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    """Identify all of the qubits that are measured in the final moment.

    Args:
        circuit: The circuit to inspect.

    Returns:
        The qubits that are measured terminally.
    """
    measure_qubits = set()
    for op in circuit[-1].operations:
        if cirq.is_measurement(op) and not isinstance(op, cirq.CircuitOperation):
            for q in op.qubits:
                measure_qubits.add(q)
    return measure_qubits


def identify_data_qubits(circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    """Identify the data qubits in a QEC circuit.

    Args:
        circuit: The circuit to inspect.

    Returns:
        The data qubits
    """
    return identify_terminal_measured_qubits(circuit).difference(
        identify_measure_qubits(circuit)
    )


def add_sweep_bits(
    circuit: cirq.Circuit, rng: np.random.Generator | None = None
) -> cirq.Circuit:
    """Add random X gates on data qubits at the beginning of a QEC circuit.

    If the first moment contains a single wait gate, then insert the X gates after it.

    Args:
        circuit: The circuit to add the gates to.
        rng: A pseudorandom number generator.

    Returns:
        The modified circuit.
    """
    rng = rng or np.random.default_rng()
    data_qubits = sorted(identify_data_qubits(circuit))
    include = rng.random(len(data_qubits))
    qubits_to_flip = np.array(data_qubits)[include > 0.5]
    idx = int(
        isinstance(circuit[0].operations[0].gate, cirq.WaitGate)
        and len(circuit[0].operations) == 1
    )
    return circuit[:idx] + cirq.X.on_each(qubits_to_flip) + circuit[idx:]
