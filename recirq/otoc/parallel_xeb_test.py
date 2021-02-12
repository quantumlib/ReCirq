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

import cirq
import numpy as np

from recirq.otoc import build_xeb_circuits, parallel_xeb_fidelities


# Simulate 3-qubit parallel XEB using Cirq.Simulator() and ensure fidelities are close to 1.
def test_parallel_xeb_fidelities() -> None:
    sampler = cirq.Simulator()
    qubit_locs = [(0, 0), (0, 1), (0, 2)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]

    int_layers = [{(qubit_locs[0], qubit_locs[1])}, {(qubit_locs[1], qubit_locs[2])}]

    xeb_configs = [
        [cirq.Moment([cirq.ISWAP(qubits[0], qubits[1]) ** 0.5])],
        [cirq.Moment([cirq.ISWAP(qubits[1], qubits[2]) ** 0.5])],
    ]

    num_circuits = 10
    num_num_cycles = range(3, 23, 5)
    num_cycles = len(num_num_cycles)

    all_bits = []
    all_sq_gates = []
    for xeb_config in xeb_configs:
        bits = []
        sq_gates = []
        for i in range(num_circuits):
            circuits, sq_gate_indices_i = build_xeb_circuits(
                qubits, num_num_cycles, xeb_config, random_seed=i
            )
            sq_gates.append(sq_gate_indices_i)
            for c in circuits:
                c.append(cirq.measure(*qubits, key="z"))
            sweep_params = [{} for _ in range(len(circuits))]
            job = sampler.run_batch(programs=circuits, params_list=sweep_params, repetitions=5000)
            bits.append([job[j][0].measurements["z"] for j in range(num_cycles)])
        all_bits.append(bits)
        all_sq_gates.append(sq_gates)

    fsim_angles_init = {
        "theta": -0.25 * np.pi,
        "delta_plus": 0,
        "delta_minus_off_diag": 0,
        "delta_minus_diag": 0,
        "phi": 0.0,
    }
    xeb_results = parallel_xeb_fidelities(
        qubit_locs,
        num_num_cycles,
        all_bits,
        all_sq_gates,
        fsim_angles_init,
        interaction_sequence=int_layers,
        gate_to_fit="sqrt-iswap",
        num_restarts=1,
        num_points=4,
        print_fitting_progress=False,
    )

    f_01 = xeb_results.raw_data[(qubit_locs[0], qubit_locs[1])].fidelity_unoptimized[1]
    f_12 = xeb_results.raw_data[(qubit_locs[1], qubit_locs[2])].fidelity_unoptimized[1]

    assert np.allclose(f_01, 1, atol=0.1)
    assert np.allclose(f_12, 1, atol=0.1)
