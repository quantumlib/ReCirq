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

"""Example for measuring OTOC of several random circuits."""

import cirq
import cirq_google as cg
import numpy as np
from matplotlib import pyplot as plt

from recirq.otoc.otoc_circuits import build_otoc_circuits


def main():
    # Set the random seed for the OTOC circuits.
    np.random.seed(0)

    # Specify project ID and processor name.
    processor_name = "rainbow"
    sampler = cg.get_engine_sampler(processor_id=processor_name, gate_set_name="fsim")

    # Specify qubits to measure. The qubits must form a line. The ancilla qubit
    # must be connected to the first qubit.
    qubit_locs = [(3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5), (0, 6), (1, 6)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
    ancilla_loc = (3, 2)
    ancilla = cirq.GridQubit(*ancilla_loc)
    num_qubits = len(qubits)

    # Specify how the qubits interact. For the 1D chain in this example, only two
    # layers of two-qubit gates (CZ is used here) is needed to enact all
    # interactions.
    int_sets = [
        {(qubits[i], qubits[i + 1]) for i in range(0, num_qubits - 1, 2)},
        {(qubits[i], qubits[i + 1]) for i in range(1, num_qubits - 1, 2)},
    ]

    forward_ops = {
        (qubit_locs[i], qubit_locs[i + 1]): cirq.Circuit([cirq.CZ(qubits[i], qubits[i + 1])])
        for i in range(num_qubits - 1)
    }

    reverse_ops = forward_ops

    # Build two random circuit instances (each having 12 cycles).
    circuit_list = []
    cycles = range(12)
    num_trials = 2
    for i in range(num_trials):
        rand_nums = np.random.choice(8, (num_qubits, max(cycles)))
        circuits_i = []
        for cycle in cycles:
            circuits_ic = []
            for k, q_b in enumerate(qubits[1:]):
                circs = build_otoc_circuits(
                    qubits,
                    ancilla,
                    cycle,
                    int_sets,
                    forward_ops=forward_ops,
                    reverse_ops=reverse_ops,
                    butterfly_qubits=q_b,
                    cycles_per_echo=2,
                    sq_gates=rand_nums,
                    use_physical_cz=True,
                )
                # If q_b is the measurement qubit, add both the normalization circuits and the
                # circuits with X as the butterfly operator. Otherwise only add circuits with Y
                # being the butterfly operator.
                if k == 0:
                    circuits_ic.extend(circs.butterfly_I)
                    circuits_ic.extend(circs.butterfly_X)
                else:
                    circuits_ic.extend(circs.butterfly_X)
            circuits_i.append(circuits_ic)
        circuit_list.append(circuits_i)

    # Measure the OTOCs of the two random circuits.
    results = []
    for i, circuits_i in enumerate(circuit_list):
        results_i = np.zeros((num_qubits, len(cycles)))
        for c, circuits_ic in enumerate(circuits_i):
            print("Measuring circuit instance {}, cycle {}...".format(i, c))
            stats = int(2000 + 10000 * (c / max(cycles)) ** 3)
            params = [{} for _ in range(len(circuits_ic))]
            job = sampler.run_batch(programs=circuits_ic, params_list=params, repetitions=stats)
            for d in range(num_qubits):
                p = np.mean(job[4 * d][0].measurements["z"])
                p -= np.mean(job[4 * d + 1][0].measurements["z"])
                p -= np.mean(job[4 * d + 2][0].measurements["z"])
                p += np.mean(job[4 * d + 3][0].measurements["z"])
                results_i[d, c] = 0.5 * p
        results.append(results_i)

    # Average the data for the two random circuits and plot out results.
    results_ave = np.mean(results, axis=0)
    fig = plt.figure()
    plt.plot(results_ave[0, :], "ko-", label="Normalization")
    for i in range(1, num_qubits):
        plt.plot(results_ave[i, :], "o-", label="Qubit {}".format(i + 1))
    plt.xlabel("Cycles")
    plt.ylabel(r"$\langle \sigma_y \rangle$")
    plt.legend()


if __name__ == "__main__":
    main()
