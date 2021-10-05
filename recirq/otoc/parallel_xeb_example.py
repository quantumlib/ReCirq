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

"""Example for performing parallel-XEB and calibrating unitary shifts."""
import os

import cirq
import cirq_google as cg
import numpy as np

from recirq.otoc.parallel_xeb import build_xeb_circuits, parallel_xeb_fidelities, plot_xeb_results
from recirq.otoc.utils import save_data


def main():
    # Specify a working directory, project ID and processor name.
    dir_str = os.getcwd()
    processor_name = "rainbow"
    sampler = cg.get_engine_sampler(processor_id=processor_name, gate_set_name="fsim")

    # Specify qubits to measure. Here we choose the qubits to be on a line.
    qubit_locs = [(3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5), (0, 6), (1, 6)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
    num_qubits = len(qubits)

    # Specify the gate layers to calibrate. For qubits on a line, all two-qubit
    # gates can be calibrated in two layers.
    int_layers = [
        {(qubit_locs[i], qubit_locs[i + 1]) for i in range(0, num_qubits - 1, 2)},
        {(qubit_locs[i], qubit_locs[i + 1]) for i in range(1, num_qubits - 1, 2)},
    ]

    xeb_configs = [
        [
            cirq.Moment(
                [cirq.ISWAP(qubits[i], qubits[i + 1]) ** 0.5 for i in range(0, num_qubits - 1, 2)]
            )
        ],
        [
            cirq.Moment(
                [cirq.ISWAP(qubits[i], qubits[i + 1]) ** 0.5 for i in range(1, num_qubits - 1, 2)]
            )
        ],
    ]

    # Specify the number of random circuits in parallel XEB the cycles at which
    # fidelities are to be measured.
    num_circuits = 10
    num_num_cycles = range(3, 75, 10)
    num_cycles = len(num_num_cycles)

    # Generate random XEB circuits, measure the resulting bit-strings, and save
    # the data as well as random circuit information.
    all_bits = []
    all_sq_gates = []
    for xeb_config in xeb_configs:
        bits = []
        sq_gates = []
        for i in range(num_circuits):
            print(i)
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

    # Perform fits on each qubit pair to miminize the cycle errors. The fitting
    # results are saved to the working directory.
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
        num_restarts=5,
    )

    plot_xeb_results(xeb_results)

    save_data(xeb_results.correction_gates, dir_str + "/gate_corrections")


if __name__ == "__main__":
    main()
