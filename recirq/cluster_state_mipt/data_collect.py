# Copyright 2025 Google
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

"""Data collection script for cluster state measurements.

This script collects measurement data for cluster states using randomized compiling.
"""

import os
import cirq
import cirq_google as cg
import numpy as np
from recirq.cluster_state_mipt import get_circuit, get_two_qubits_6x6


def collect_data(
    engine: cg.Engine,
    processor_id: str,
    device_config_name: str = 'processor_config',
    repetitions: int = 10**6,
    num_rc_circuits: int = 20,
    folder_name: str = 'experiment_data',
) -> None:
    """Collect measurement data for cluster states.

    Args:
        engine: The Cirq engine to use for execution.
        processor_id: The ID of the processor to use.
        device_config_name: The device configuration name.
        repetitions: Number of repetitions for each circuit.
        num_rc_circuits: Number of randomized compiling circuits.
        folder_name: Name of the folder to save data in.

    Raises:
        ValueError: If repetitions or num_rc_circuits is less than or equal to 0.
    """
    if repetitions <= 0:
        raise ValueError('repetitions must be greater than 0')
    if num_rc_circuits <= 0:
        raise ValueError('num_rc_circuits must be greater than 0')

    sampler_real = engine.get_processor(processor_id).get_sampler(
        device_config_name=device_config_name)
    sampler = sampler_real
    theta_range = np.linspace(0, np.pi/2, 11)
    phi = np.pi*(5/4)

    for d in [3,4,5,6]:
        qubits_matrix, probe_qubits, anc_pairs, all_qubits = get_two_qubits_6x6(d=d)
        for theta_idx in range(len(theta_range)):
            for iteration in range(5):
                for b1 in range(3):
                    for b2 in range(3):
                        circ = get_circuit(
                            qubits_matrix,
                            theta=theta_range[theta_idx],
                            phi=phi,
                            probe_qubits=probe_qubits,
                            basis=[b1, b2],
                            anc_pairs=anc_pairs)
                        circ.append(cirq.measure(*all_qubits, key='m'))
                        circuit_batch = [
                            cirq.transformers.gauge_compiling.CZGaugeTransformer(circ)
                            for _ in range(num_rc_circuits)]
                        circuit_batch = [
                            cirq.merge_single_qubit_moments_to_phxz(circuit)
                            for circuit in circuit_batch]
                        result = sampler.run_batch(
                            circuit_batch,
                            repetitions=int(repetitions/num_rc_circuits))
                        print(f'distance={d}, loop={iteration}, theta={theta_idx}, basis={b1},{b2}')
                        
                        # Convert measurements to numpy arrays and concatenate
                        measurements = np.concatenate([
                            np.array(r[0].measurements['m']).astype(np.int32)
                            for r in result])
                        
                        # Create directory if it doesn't exist
                        save_dir = f'data/{folder_name}_d={d}/theta{theta_idx}/loop{iteration}'
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Save data using numpy
                        np.save(
                            f'{save_dir}/theta={theta_idx}_({b1},{b2}).npy',
                            measurements)


def main() -> None:
    """Main function to run the data collection."""
    # Replace with your GCP project ID where the quantum processors are registered
    engine = cg.get_engine('your-gcp-project-id')
    collect_data(engine=engine, processor_id='your-processor-id')


if __name__ == '__main__':
    main()
