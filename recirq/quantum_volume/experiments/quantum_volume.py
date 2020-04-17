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

import os
from typing import Optional, List

import numpy as np

import cirq
from cirq.contrib.quantum_volume import quantum_volume

import recirq


@recirq.json_serializable_dataclass(namespace='recirq.quantum_volume',
                                    registry=recirq.Registry,
                                    frozen=True)
class QuantumVolumeTask:
    """Script to run the quantum volume algorithm on a device, per the benchmark defined by IBM in
    https://arxiv.org/abs/1811.12926.

    This will run the Quantum Volume benchmark on the Sycamore device on the given depths (every combination of
    depths 2 through 6 by default) with 100 repetitions each depth. You can configure these parameters by editing
    this script directly. All the business logic is deferred to a Quantum Volume back in public Cirq - this file
    exists primarily to pull in devices that we are not yet ready to release.

    Attributes:
        dataset_id: A unique identifier for this dataset.
        device_name: The device to run on, by name.
        num_qubits: The number of qubits in the generated circuits.
        qubits: All of the possible qubits that the algorithm can run on. If empty, 
            we will use any qubits on the device.
        n_shots: The number of repetitions for each circuit.
        n_circuits: The number of circuits to run the algorithm.
        depth: The number of layers to generate in the circuit.
    """
    dataset_id: str
    device_name: str
    n_qubits: int
    n_shots: int
    n_circuits: int
    depth: int
    readout_error_correction: bool
    qubits: Optional[List[cirq.GridQubit]] = None

    @property
    def fn(self):
        n_shots = _abbrev_n_shots(self.n_shots)
        qubits = _abbrev_grid_qubits(self.qubits) + '/'

        return (f'{self.dataset_id}/'
                f'{self.device_name}/'
                f'{qubits}/'
                f'{self.n_circuits}_{self.depth}_{self.n_qubits}_{n_shots}')


# Define the following helper functions to make nicer `fn` keys
# for the tasks:


def _abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)


def _abbrev_grid_qubits(qubits: Optional[List[cirq.GridQubit]]) -> str:
    """Formatted a list of grid qubits component of a filename"""
    if not qubits:
        return ''
    return '-'.join([f'{qubit.row}_{qubit.col}' for qubit in qubits])


EXPERIMENT_NAME = 'quantum-volume'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


def run_quantum_volume(task: QuantumVolumeTask, base_dir=None):
    """Execute a :py:class:`QuantumVolumeTask` task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task} already exists. Skipping.")
        return

    sampler = recirq.get_sampler_by_name(device_name=task.device_name)
    device = recirq.get_device_obj_by_name(device_name=task.device_name)
    device_or_qubits = task.qubits if task.qubits else device

    # Run the jobs
    print("Collecting data", flush=True)
    results = quantum_volume.calculate_quantum_volume(
        num_qubits=task.n_qubits,
        depth=task.depth,
        num_circuits=task.n_circuits,
        device_or_qubits=device_or_qubits,
        samplers=[sampler],
        repetitions=task.n_shots,
        random_state=np.random.RandomState(
            int(hash(task.dataset_id)) % (2**32 - 1)),
        compiler=lambda c: cirq.google.optimized_for_sycamore(
            c,
            new_device=device,
            optimizer_type='sycamore',
            tabulation_resolution=0.008),
        add_readout_error_correction=task.readout_error_correction)
    # Save the results CODE REVIEW QUESTION: This attempts to serialize a QuantumVolumeResult, which contains a
    # Circuit, which contains a SerializableDevice, which is not JSON serializable. What's the best way to resolve
    # this?
    recirq.save(task=task, data={
        'results': results,
    }, base_dir=base_dir)
