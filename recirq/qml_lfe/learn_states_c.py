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

"""Draw I + P pauli product data using shadow strategy.

learn_states_c.py --n=6 --n_paulis=20 --n_sweeps=500 --n_shots=1000

Will create 20 random paulistring circuits on 6 qubits.
For each paulistring circuit, payloads containing n_sweeps sweeps are
executed until n_shots total samples have been drawn from that circuit.
By default, the bitstring data will be saved in the data folder. One can
also set `use_engine` to True in order to run this against a processor on
quantum engine.
"""
from typing import Dict, List, Tuple

import os
import cirq
import numpy as np
import sympy

from absl import app
from absl import logging

from recirq.qml_lfe import circuit_blocks
from recirq.qml_lfe import run_config


def _create_basis_sweeps(
    H_params: List[sympy.Symbol],
    S_params: List[sympy.Symbol],
    n_shots: int,
    rand_state: np.random.RandomState,
) -> Tuple[List[Dict[str, int]], np.ndarray]:
    """Generate sweeps that transform to many different random pauli bases."""
    assert len(H_params) == len(S_params)
    all_sweeps = []
    all_bases = rand_state.randint(0, 3, size=(n_shots, len(H_params)))
    for r in range(n_shots):
        basis = all_bases[r]
        sweep = dict()
        for i in range(len(H_params)):
            if basis[i] == 0:
                # Identity.
                sweep[H_params[i]] = 0.0
                sweep[S_params[i]] = 0.0
            elif basis[i] == 1:
                # H.
                sweep[H_params[i]] = 0.5
                sweep[S_params[i]] = 0.5
            elif basis[i] == 2:
                # HS.
                sweep[H_params[i]] = -1.0
                sweep[S_params[i]] = 0.5

        all_sweeps.append(sweep)

    return all_sweeps, all_bases


def build_circuit(
    qubit_pairs: List[List[cirq.Qid]],
    pauli: str,
    n_shots: int,
    rand_state: np.random.RandomState,
) -> Tuple[cirq.Circuit, List[Dict[str, int]], np.ndarray]:
    """Create I + P problem circuit using shadows (not two copy).

    Args:
        qubit_pairs: List of qubit pairs.
        pauli: Python str containing characters 'I', 'X', 'Y' or 'Z'.
        n_shots: Number of repetitions to generate for sweeps.
        rand_state: np.random.RandomState source of randomness.

    Returns:
        A (circuit, sweep, basis) tuple, runnable using `run_sweep`.
    """
    a_qubits = [pair[0] for pair in qubit_pairs]
    b_qubits = [pair[1] for pair in qubit_pairs]
    all_qubits = np.concatenate(qubit_pairs)

    flip_params = sympy.symbols(f"param_0:{len(qubit_pairs) * 2}")
    S_params = sympy.symbols(f"Sparam_0:{len(qubit_pairs)}")
    H_params = sympy.symbols(f"Hparam_0:{len(qubit_pairs)}")

    # Add X flips.
    ret_circuit = cirq.Circuit(cirq.X(q) ** p for q, p in zip(all_qubits, flip_params))

    # Add basis turns a and b.
    ret_circuit += [
        circuit_blocks.inv_z_basis_gate(p)(q) for q, p in zip(a_qubits, pauli)
    ]
    ret_circuit += [
        circuit_blocks.inv_z_basis_gate(p)(q) for q, p in zip(b_qubits, pauli)
    ]

    # Turn for X/Y, p is either 0 or 1
    ret_circuit += cirq.Circuit(
        cirq.PhasedXZGate(axis_phase_exponent=p, x_exponent=r, z_exponent=0.0)(q)
        for q, (p, r) in zip(a_qubits, zip(H_params, S_params))
    )

    # Add measurements.
    for i, qubit in enumerate(a_qubits):
        ret_circuit += cirq.measure(qubit, key="q{}".format(i))

    # Merge single qubit operations, flatten moments and align measurements.
    ret_circuit = cirq.merge_single_qubit_gates_to_phxz(ret_circuit)
    ret_circuit = cirq.drop_empty_moments(ret_circuit)
    ret_circuit = run_config.flatten_circuit(ret_circuit)
    ret_circuit = cirq.synchronize_terminal_measurements(ret_circuit)

    # Create randomized flippings. These flippings will contain values of 1,0.
    # which will turn the X gates on or off.
    params = circuit_blocks.create_randomized_sweeps(
        pauli, flip_params, n_shots, rand_state
    )

    # Choose between Z,X,Y basis measurement basis.
    basis_sweeps, basis_arr = _create_basis_sweeps(
        H_params, S_params, n_shots, rand_state
    )

    all_params = [{**x, **y} for x, y in zip(params, basis_sweeps)]
    logging.debug(
        f"Generated circuit w/ depth {len(ret_circuit)} and {len(params)} sweeps."
    )
    return ret_circuit, all_params, basis_arr


def run_and_save(
    n: int,
    n_paulis: int,
    n_sweeps: int,
    n_shots: int,
    save_dir: str,
    use_engine: bool,
) -> None:
    """Run and save bitstring data for I + P experiment w/ twocopy.

    Note: uses qubit layouts native to a Sycamore device (Weber).

    Args:
        n: Number of system qubits to use (total qubits == 2 * n).
        n_paulis: Number of pauli circuits to generate.
        n_sweeps: The number of circuits to send off for execution in
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
    rand_source = np.random.RandomState(1234)

    logging.info("Generating pauli strings.")
    paulis = np.array(["X", "Y", "Z", "I"])
    pauli_strings = rand_source.choice(a=paulis, size=(n_paulis, n), replace=True)

    for pauli in pauli_strings:
        logging.info(f"Processing pauli: {pauli}")
        circuit, sweeps, basis_arr = build_circuit(
            system_pairs, pauli, n_shots, rand_source
        )

        all_results = []
        for b in range(0, n_shots, n_sweeps):
            results = run_config.execute_sweep(
                circuit, sweeps[b : b + n_sweeps], use_engine
            )

            batch_results = []
            for j, single_circuit_samples in enumerate(results):
                qubit_order = [f"q{i}" for i in range(n)]
                out0 = single_circuit_samples.data[qubit_order].to_numpy()
                batch_results.append(np.squeeze(out0))

            batch_results = np.array(batch_results)
            all_results.append(batch_results)

        all_results = np.concatenate(all_results)
        file_name = "Q-size-{}-pauli-{}".format(n, "".join(t for t in pauli))
        basis_file_name = "Q-size-{}-pauli-{}-basis".format(
            n, "".join(t for t in pauli)
        )
        np.save(os.path.join(save_dir, file_name), all_results)  # [n_shots, 2 * n]
        np.save(os.path.join(save_dir, basis_file_name), basis_arr)
        logging.debug("Saved: " + file_name)


def main(_):
    run_and_save(
        state_flags.FLAGS.n,
        state_flags.FLAGS.n_paulis,
        state_flags.FLAGS.n_sweeps,
        state_flags.FLAGS.n_shots,
        state_flags.FLAGS.save_dir,
        state_flags.FLAGS.use_engine,
    )


if __name__ == "__main__":
    from recirq.qml_lfe import state_flags

    app.run(main)
