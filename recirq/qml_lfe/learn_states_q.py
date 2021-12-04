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

"""Draw I + P pauli product data using two copy strategy.
style circuits.

learn_states_q.py --n=6 --n_paulis=20 --batch_size=500 --n_shots=1000

Will create paulistring circuits all on 12 qubits (two systems of size 6).
Once the circuits are generated, batch_size sweeps and a single circuit at
a timewill be sent will be sent for simulation/execution, drawing n_shots
total samples from each one using `run_sweep` in Cirq. By default the
bitstring data will be saved in the data folder. One can also set
`use_engine` to True in order to run this against a processor on
quantum engine.
"""
from typing import Dict, List

import os
import cirq
import numpy as np
import sympy
from . import circuit_blocks
from . import run_config

from absl import app
from absl import logging


def _create_randomized_sweeps(
    hidden_p: str, symbols: Dict[str, int], num_reps: int
) -> List[Dict[str, int]]:
    last_i = 0
    for i, pauli in enumerate(hidden_p):
        if pauli != "I":
            last_i = i

    sign_p = -1 if np.random.random() < 0.5 else 1
    all_sweeps = []
    for _ in range(num_reps):
        current_sweep = dict()
        for twocopy in [0, 1]:
            parity = sign_p * (1 if np.random.random() <= 0.95 else -1)
            for i, pauli in enumerate(hidden_p):
                current_symbol = symbols[2 * i + twocopy]
                current_sweep[current_symbol] = 0
                if pauli != "I":
                    if last_i == i:
                        if parity == -1:
                            current_sweep[current_symbol] = 1
                    elif np.random.choice([0, 1]) == 1:
                        parity *= -1
                        current_sweep[current_symbol] = 1
                else:
                    if np.random.choice([0, 1]) == 1:
                        current_sweep[current_symbol] = 1
        all_sweeps.append(current_sweep)

    return all_sweeps


def build_circuit(qubit_pairs: List[List[cirq.Qid]], pauli: str, num_reps: int):
    """Create I + P problem circuit between qubit pairs.

    Args:
        qubit_pairs: List of qubit pairs.
        pauli: Python str containing characters 'I', 'X', 'Y' or 'Z'.
        num_reps: Number of repetitions to generate for sweeps.

    Returns:
        A (circuit, sweep) tuple, runnable using `run_sweep`.
    """
    a_qubits = [pair[0] for pair in qubit_pairs]
    b_qubits = [pair[1] for pair in qubit_pairs]
    all_qubits = np.concatenate(qubit_pairs)

    flip_params = sympy.symbols(f"param_0:{len(qubit_pairs) * 2}")

    # Add X flips.
    ret_circuit = cirq.Circuit(cirq.X(q) ** p for q, p in zip(all_qubits, flip_params))

    # Add basis turns a and b.
    ret_circuit += [
        circuit_blocks.inv_z_basis_gate(p)(q) for q, p in zip(a_qubits, pauli)
    ]
    ret_circuit += [
        circuit_blocks.inv_z_basis_gate(p)(q) for q, p in zip(b_qubits, pauli)
    ]

    # Add un-bell pair.
    ret_circuit += [circuit_blocks.un_bell_pair_block(pair) for pair in qubit_pairs]

    # Add measurements.
    for i, qubit in enumerate(all_qubits):
        ret_circuit += cirq.measure(qubit, key=f"q{i}")

    # Merge single qubit operations, flatten moments and align measurements.
    cirq.merge_single_qubit_gates_into_phxz(ret_circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit=ret_circuit)
    ret_circuit = run_config.flatten_circuit(ret_circuit)
    cirq.SynchronizeTerminalMeasurements().optimize_circuit(circuit=ret_circuit)

    # Create randomized flippings. These flippings will contain values of 1,0.
    # which will turn the X gates on or off.
    params = _create_randomized_sweeps(pauli, flip_params, num_reps)
    logging.debug(
        f"Generated circuit w/ depth {len(ret_circuit)} and {len(params)} sweeps."
    )
    return ret_circuit, params


def run_and_save(
    n: int,
    n_paulis: int,
    batch_size: int,
    n_shots: int,
    save_dir: str,
    use_engine: bool,
) -> None:
    """Run and save bitstring data for I + P experiment w/ twocopy.

    Note: uses qubit layouts native to a Sycamore device (Weber).

    Args:
        n: Number of system qubits to use (total qubits == 2 * n).
        n_paulis: Number of pauli circuits to generate.
        batch_size: The number of circuits to send off for execution in
            a single payload (Does not affect experimental resluts).
        n_shots: Number of shots to draw from each circuit.
        save_dir: str or Path to directory where data is saved.
        use_engine: Whether or not to make use of quantum engine and the
            weber processor. Note this requires the GOOGLE_CLOUD_PROJECT
            environment variable to be set, along with the required cloud
            permissions.
    """
    logging.info("Beginning quantum-enhanced circuit generation.")
    system_pairs = run_config.qubit_pairs()
    system_pairs = system_pairs[:n]

    logging.info("Generating pauli strings.")
    paulis = np.array(["X", "Y", "Z", "I"])
    pauli_strings = np.random.choice(a=paulis, size=(n_paulis, n), replace=True)

    for pauli in pauli_strings:
        logging.info(f"Processing pauli: {pauli}")
        circuit, sweeps = build_circuit(system_pairs, pauli, n_shots)

        all_results = []
        for b in range(0, n_shots, batch_size):
            results = run_config.execute_sweep(
                circuit, sweeps[b : b + batch_size], use_engine
            )

            batch_results = []
            for j, single_circuit_samples in enumerate(results):
                qubit_order = [f"q{i}" for i in range(2 * n)]
                out0 = single_circuit_samples.data[qubit_order].to_numpy()
                batch_results.append(np.squeeze(out0))

            batch_results = np.array(batch_results)
            all_results.append(batch_results)

        all_results = np.concatenate(all_results)
        name0 = "Q-size-{}-pauli-{}".format(n, "".join(t for t in pauli))
        np.save(os.path.join(save_dir, name0), all_results)  # [n_shots, 2 * n]
        logging.debug("Saved: " + name0)


def main(_):
    from . import state_flags

    run_and_save(
        state_flags.n,
        state_flags.n_paulis,
        state_flags.batch_size,
        state_flags.n_shots,
        state_flags.save_dir,
        state_flags.use_engine,
    )


if __name__ == "__main__":
    app.run(main)
