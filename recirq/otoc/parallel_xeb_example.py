import cirq
import numpy as np

from recirq.otoc.parallel_xeb import build_xeb_circuits, parallel_xeb_fidelities
from recirq.otoc.utils import cz_to_sqrt_iswap, save_data

dir_str = '/usr/local/google/home/mixiao/PycharmProjects/ReCirq/recirq/otoc' \
          '/data'
project_id = 'xiaomidec2018'
engine = cirq.google.Engine(project_id=project_id,
                            proto_version=cirq.google.ProtoVersion.V2)
processor_name = 'mcgee'

qubit_locs = [(3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5), (0, 6), (1, 6)]
qubits = [cirq.GridQubit(*idx) for idx in qubit_locs]
ancilla_loc = (4, 3)
ancilla = cirq.GridQubit(*ancilla_loc)
num_qubits = len(qubits)

int_layers = [
    {(qubit_locs[i], qubit_locs[i + 1]) for i in range(0, num_qubits - 1, 2)},
    {(qubit_locs[i], qubit_locs[i + 1]) for i in range(1, num_qubits - 1, 2)}]

xeb_configs = [[cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** 0.5
                             for i in range(0, num_qubits - 1, 2)])],
               [cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** 0.5
                             for i in range(1, num_qubits - 1, 2)])],
               [cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** -0.5
                             for i in range(0, num_qubits - 1, 2)])],
               [cirq.Moment([cirq.ISWAP(qubits[i], qubits[i + 1]) ** -0.5
                             for i in range(1, num_qubits - 1, 2)])],
               [cz_to_sqrt_iswap(ancilla, qubits[0])]]

num_trials = 50
num_cycle_range = range(3, 75, 10)
num_cycles = len(num_cycle_range)

all_bits_sqrtiswap = []
all_phases_sqrtiswap = []
for xeb_config in xeb_configs[0:2]:
    bits_sqrtiswap = []
    phases_sqrtiswap = []
    for i in range(num_trials):
        print(i)
        circuits, rand_indices = build_xeb_circuits(
            qubits, num_cycle_range, xeb_config, random_seed=i)
        phases_sqrtiswap.append(rand_indices)
        for c in circuits:
            c.append(cirq.measure(*qubits, key='z'))
        sweep_params = [{} for _ in range(len(circuits))]
        job = engine.run_batch(
            programs=circuits,
            params_list=sweep_params,
            repetitions=5000,
            processor_ids=[processor_name],
            gate_set=cirq.google.SQRT_ISWAP_GATESET)
        bits_sqrtiswap.append([job.results()[j].measurements['z'] for j in
                               range(num_cycles)])
    all_bits_sqrtiswap.append(bits_sqrtiswap)
    all_phases_sqrtiswap.append(phases_sqrtiswap)

fsim_angles_0 = {'theta': -0.25 * np.pi, 'delta_plus': 0,
                 'delta_minus_off_diag': 0, 'delta_minus_diag': 0,
                 'phi': 0.0}
(fitted_gates_0, corrected_gates_0, fsim_fitted_0, err_0_opt,
 err_0_unopt) = parallel_xeb_fidelities(
    qubit_locs, num_cycle_range, all_bits_sqrtiswap,
    all_phases_sqrtiswap, fsim_angles_0,
    interaction_sequence=int_layers, gate_to_fit='sqrt-iswap',
    plot_individual_traces=True, num_restarts=5,
    plot_histograms=True, save_directory=None)

save_data(corrected_gates_0, dir_str + '/gate_corrections')