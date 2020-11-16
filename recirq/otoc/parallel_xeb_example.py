import os

import cirq
import random
import string

from cirq.experiments import build_entangling_layers

from recirq.otoc.parallel_xeb import build_xeb_circuits
from recirq.otoc.utils import cz_to_sqrt_iswap, save_data

os.chdir('/usr/local/google/home/mixiao/PycharmProjects/ReCirq/recirq/otoc')
dir_str = '/data'

project_id = 'xiaomidec2018'
engine = cirq.google.Engine(project_id=project_id,
                            proto_version=cirq.google.ProtoVersion.V2)

qubit_locs = [(3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5), (0, 6), (1, 6)]
qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
ancilla = cirq.GridQubit(4, 3)
num_qubits = len(qubits)

xeb_configs = [cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** 0.5
                            for i in range(0, num_qubits - 1, 2)]),
               cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** 0.5
                            for i in range(1, num_qubits - 1, 2)]),
               cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** -0.5
                            for i in range(0, num_qubits - 1, 2)]),
               cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** -0.5
                            for i in range(1, num_qubits - 1, 2)]),
cz_to_sqrt_iswap(ancilla, qubits[0])
               ]

ent_layers = build_entangling_layers(qubits, cirq.ISWAP ** -0.5)
ent_layers.extend(build_entangling_layers(qubits, cirq.ISWAP ** 0.5))
ent_layers.append(cirq.Circuit(cz_to_sqrt_iswap(ancilla, qubits[0])))

layers = [[l] for l in ent_layers]
num_trials = 50
num_cycle_range = range(3, 75, 10)
save_data(
    num_cycle_range, '{}/xeb_cycle_numbers_{}_tq'.format(dir_str, date_str))

for l in range(0, 9):
    all_circuits_l = []
    all_bits_l = []
    for i in range(num_trials):
        print(l, i)
        if l < 8:
            circuits, rand_indices = build_xeb_circuits(
                qubits, num_cycle_range, layers[l], random_seed=i)
        else:
            circuits, rand_indices = build_xeb_circuits(
                [ancilla, qubits[0]], num_cycle_range, layers[l], random_seed=i)

        all_circuits_l.append(rand_indices)
        bits_il = []
        for circuit in circuits:
            # print(circuit)
            if l < 8:
                circuit.append(cirq.measure(*qubits, key='z'))
            else:
                circuit.append(cirq.measure(*([ancilla, qubits[0]]), key='z'))
            name = 'example-%s' % ''.join(random.choice(
                string.ascii_uppercase + string.digits) for _ in range(8))
            job = engine.run_sweep(
                program=circuit,
                program_id=name,
                repetitions=5000,
                processor_ids=['rainbow'],
                gate_set=cirq.google.SQRT_ISWAP_GATESET)
            bits_0 = job.results()[0].measurements['z']
            bits_il.append(bits_0)
        all_bits_l.append(bits_il)

    save_data(all_circuits_l, '{}/xeb_random_indices_for_'
                              'circuits_{}_{}'.format(dir_str, date_str, l))
    save_data(
        all_bits_l, '{}/xeb_bit_strings_{}_{}'.format(dir_str, date_str, l))
