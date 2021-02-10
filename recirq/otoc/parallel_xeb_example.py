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

import cirq
import numpy as np

from recirq.otoc.parallel_xeb import build_xeb_circuits, parallel_xeb_fidelities
from recirq.otoc.utils import save_data


def main():
    # Specify a working directory, project ID and processor name.
    dir_str = ""
    project_id = ""
    engine = cirq.google.Engine(project_id=project_id, proto_version=cirq.google.ProtoVersion.V2)
    processor_name = "rainbow"

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
    num_trials = 20
    num_cycle_range = range(3, 75, 10)
    num_cycles = len(num_cycle_range)

    # Generate random XEB circuits, measure the resulting bit-strings, and save
    # the data as well as random circuit information.
    all_bits_sqrtiswap = []
    all_phases_sqrtiswap = []
    for xeb_config in xeb_configs:
        bits_sqrtiswap = []
        phases_sqrtiswap = []
        for i in range(num_trials):
            print(i)
            circuits, rand_indices = build_xeb_circuits(
                qubits, num_cycle_range, xeb_config, random_seed=i
            )
            phases_sqrtiswap.append(rand_indices)
            for c in circuits:
                c.append(cirq.measure(*qubits, key="z"))
            sweep_params = [{} for _ in range(len(circuits))]
            job = engine.run_batch(
                programs=circuits,
                params_list=sweep_params,
                repetitions=5000,
                processor_ids=[processor_name],
                gate_set=cirq.google.SQRT_ISWAP_GATESET,
            )
            bits_sqrtiswap.append([job.results()[j].measurements["z"] for j in range(num_cycles)])
        all_bits_sqrtiswap.append(bits_sqrtiswap)
        all_phases_sqrtiswap.append(phases_sqrtiswap)

    # Perform fits on each qubit pair to miminize the cycle errors. The fitting
    # results are saved to the working directory.
    fsim_angles_0 = {
        "theta": -0.25 * np.pi,
        "delta_plus": 0,
        "delta_minus_off_diag": 0,
        "delta_minus_diag": 0,
        "phi": 0.0,
    }
    (
        fitted_gates_0,
        corrected_gates_0,
        fsim_fitted_0,
        err_0_opt,
        err_0_unopt,
    ) = parallel_xeb_fidelities(
        qubit_locs,
        num_cycle_range,
        all_bits_sqrtiswap,
        all_phases_sqrtiswap,
        fsim_angles_0,
        interaction_sequence=int_layers,
        gate_to_fit="sqrt-iswap",
        plot_individual_traces=True,
        num_restarts=5,
        plot_histograms=True,
        save_directory=None,
    )

    save_data(corrected_gates_0, dir_str + "/gate_corrections")


if __name__ == "__main__":
    main()
