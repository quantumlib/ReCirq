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
import numpy as np
from matplotlib import pyplot as plt

from recirq.otoc.otoc_circuits import build_otoc_circuits


def main():
    # Specify project ID and processor name.
    project_id = ''
    engine = cirq.google.Engine(project_id=project_id,
                                proto_version=cirq.google.ProtoVersion.V2)
    processor_name = 'rainbow'

    # Specify qubits to measure. The qubits must form a line. The ancilla qubit
    # must be connected to the first qubit.
    qubit_locs = [(3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5), (0, 6),
                  (1, 6)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
    ancilla_loc = (3, 2)
    ancilla = cirq.GridQubit(*ancilla_loc)
    num_qubits = len(qubits)

    # Specify how the qubit interact. For the 1D chain in this example, only two
    # layers of two-qubit gates (CZ is used here) is needed to enact all
    # interactions.
    int_sets = [
        {(qubits[i], qubits[i + 1]) for i in range(0, num_qubits - 1, 2)},
        {(qubits[i], qubits[i + 1]) for i in range(1, num_qubits - 1, 2)}]

    forward_ops = {(qubit_locs[i], qubit_locs[i + 1]): cirq.Circuit([cirq.CZ(
        qubits[i], qubits[i + 1])]) for i in range(num_qubits - 1)}

    reverse_ops = {(qubit_locs[i], qubit_locs[i + 1]): cirq.Circuit([cirq.CZ(
        qubits[i], qubits[i + 1])]) for i in range(num_qubits - 1)}

    # Build two random circuit instances (each having 12 cycles).
    circuit_list = []
    cycles = range(12)
    num_trials = 2
    for i in range(num_trials):
        np.random.seed(i)
        rand_nums = np.random.choice(8, (num_qubits, max(cycles)))
        circuits_i = []
        for cycle in cycles:
            circuits_ic = []
            for k, q_b in enumerate(qubits[1:]):
                circs = build_otoc_circuits(
                    qubits, ancilla, cycle, int_sets, forward_ops=forward_ops,
                    reverse_ops=reverse_ops, butterfly_qubits=q_b,
                    cycles_per_echo=2, sq_gates=rand_nums, use_physical_cz=True)
                if k == 0:
                    circuits_ic.extend(circs[0:8])
                else:
                    circuits_ic.extend(circs[4:8])
            circuits_i.append(circuits_ic)
        circuit_list.append(circuits_i)

    # Measure the OTOCs of the two random circuits.
    results = []
    for i, circuits_i in enumerate(circuit_list):
        results_i = np.zeros((num_qubits, len(cycles)))
        for c, circuits_ic in enumerate(circuits_i):
            print('Measuring circuit instance {}, cycle {}...'.format(i, c))
            stats = int(2000 + 10000 * (c / max(cycles)) ** 3)
            sweep_params = [{} for _ in range(len(circuits_ic))]
            job = engine.run_batch(
                programs=circuits_ic,
                params_list=sweep_params,
                repetitions=stats,
                processor_ids=[processor_name],
                gate_set=cirq.google.FSIM_GATESET)
            for d in range(num_qubits):
                p = np.mean(job.results()[4 * d].measurements['z'])
                p -= np.mean(job.results()[4 * d + 1].measurements['z'])
                p -= np.mean(job.results()[4 * d + 2].measurements['z'])
                p += np.mean(job.results()[4 * d + 3].measurements['z'])
                results_i[d, c] = 0.5 * p
            results.append(results_i)

    # Average the data for the two random circuits and plot out results.
    results_ave = np.mean(results, axis=0)
    fig = plt.figure()
    plt.plot(results_ave[0, :], 'ko-', label='Normalization')
    for i in range(1, num_qubits):
        plt.plot(results_ave[i, :], 'o-', label='Qubit {}'.format(i + 1))
    plt.xlabel('Cycles')
    plt.ylabel(r'$\langle \sigma_y \rangle$')
    plt.legend()


if __name__ == '__main__':
    main()
