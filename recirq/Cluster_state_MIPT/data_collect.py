"""Data collection script for cluster state measurements.

This script collects measurement data for cluster states using randomized compiling.
"""

import cirq
import cirq_google as cg
import torch
import numpy as np
from recirq.Cluster_state_MIPT import get_circuit, get_two_qubits_6x6


def collect_data(
    eng: cg.Engine,
    processor_id: str,
    device_config_name: str = 'processor_config',
    repetitions: int = 10**6,
    num_rc_circuits: int = 20,
    folder_name: str = 'experiment_data',
) -> None:
    """Collect measurement data for cluster states.

    Args:
        eng: The Cirq engine to use for execution.
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

    sampler_real = eng.get_processor(processor_id).get_sampler(
        device_config_name=device_config_name)
    sampler = sampler_real
    theta_range = np.linspace(0, np.pi/2, 11)
    phi = np.pi*(5/4)

    for d in [5]:
        qubits_matrix, probe_qubits, anc_pairs, all_qubits = get_two_qubits_6x6(d=d)
        for theta_idx in range(len(theta_range)):
            for i in range(5):  # iterations
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
                        print(f'distance={d}, loop={i}, theta={theta_idx}, basis={b1},{b2}')
                        measurements = torch.cat([
                            torch.tensor(r[0].measurements['m']).to(int)
                            for r in result])
                        torch.save(
                            measurements,
                            f'data/{folder_name}_d={d}/theta{theta_idx}/loop{i}/theta={theta_idx}_({b1},{b2}).pt')


def main() -> None:
    """Main function to run the data collection."""
    # Replace these with your specific engine and processor details
    eng = cg.get_engine('your-engine-id')
    collect_data(eng=eng, processor_id='your-processor-id')


if __name__ == '__main__':
    main()
