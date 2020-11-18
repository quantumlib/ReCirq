import cirq
import numpy as np
from matplotlib import pyplot as plt

from recirq.otoc.otoc_circuits import build_otoc_circuits

project_id = 'xiaomidec2018'
engine = cirq.google.Engine(project_id=project_id,
                            proto_version=cirq.google.ProtoVersion.V2)
processor_name = 'rainbow'

qubit_locs = [(3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5), (0, 6), (1, 6)]
qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
ancilla_loc = (3, 2)
ancilla = cirq.GridQubit(*ancilla_loc)
num_qubits = len(qubits)

int_sets = [{(qubits[i], qubits[i + 1]) for i in range(0, num_qubits - 1, 2)},
            {(qubits[i], qubits[i + 1]) for i in range(1, num_qubits - 1, 2)}]

forward_ops = {(qubit_locs[i], qubit_locs[i + 1]): cirq.Circuit([cirq.CZ(
    qubits[i], qubits[i + 1]) ** 1]) for i in range(num_qubits - 1)}

reverse_ops = {(qubit_locs[i], qubit_locs[i + 1]): cirq.Circuit([cirq.CZ(
    qubits[i], qubits[i + 1]) ** -1]) for i in range(num_qubits - 1)}

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

results = []
for i, circuits_i in enumerate(circuit_list):
    results_i = np.zeros((num_qubits, len(cycles)))
    for c, circuits_ic in enumerate(circuits_i):
        print('Measuring circuit instance {}, cycle {}...'.format(i, c))
        stats = int(2000 + 15000 * (c / max(cycles)) ** 3)
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

results_ave = np.mean(results, axis=0)
fig = plt.figure()
plt.plot(results_ave[0, :], 'ko-', label='Normalization')
for i in range(1, num_qubits):
    plt.plot(results_ave[i, :], 'o-', label='Qubit {}'.format(i + 1))
plt.xlabel('Cycles')
plt.ylabel(r'$\langle \sigma_y \rangle$')
plt.legend()
