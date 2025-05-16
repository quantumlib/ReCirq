"""Module for running measurement induced entanglement experiments on Google's quantum computer.

This module provides functionality for running experiments on Google's quantum computer
(Willow Pink) to study measurement induced entanglement on cluster states.
"""

from typing import List, Optional, Tuple
import cirq
import cirq_google as cg
import numpy as np
import torch
from recirq.measurement_entanglement.cluster_state import get_circuit, get_two_qubits_6x6


class QuantumExperiment:
    """Class for running quantum experiments on Google's quantum computer.
    
    This class handles the setup and execution of quantum experiments for studying
    measurement induced entanglement on cluster states.
    """
    
    def __init__(
        self,
        project_id: str = 'serious-cat-404923',
        processor_id: str = 'WLA1HHPR00V02_4A_PINK',
        device_config_name: str = 'd7',
        snapshot_id: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """Initialize the quantum experiment.
        
        Args:
            project_id: Google Cloud project ID.
            processor_id: ID of the quantum processor to use.
            device_config_name: Name of the device configuration.
            snapshot_id: Optional snapshot ID for calibration.
            run_name: Optional run name for calibration.
            
        Raises:
            ValueError: If neither snapshot_id nor run_name is provided.
        """
        if snapshot_id is None and run_name is None:
            raise ValueError("Either snapshot_id or run_name must be provided.")
            
        self.engine = cg.get_engine(project_id)
        if snapshot_id is not None:
            self.sampler = self.engine.get_processor(processor_id).get_sampler(
                device_config_name=device_config_name,
                snapshot_id=snapshot_id
            )
        else:
            self.sampler = self.engine.get_sampler(
                processor_id=processor_id,
                run_name=run_name,
                device_config_name=device_config_name
            )
        self.simulator = cirq.Simulator()
        
    def test_bell_state(self, repetitions: int = 1000) -> cirq.Result:
        """Run a test Bell state experiment.
        
        Args:
            repetitions: Number of times to run the circuit.
            
        Returns:
            The measurement results from the Bell state experiment.
        """
        qubits = [cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)]
        circuit = cirq.Circuit(
            cirq.H.on_each(*qubits),
            cirq.CZ(*qubits),
            cirq.H(qubits[1]),
            cirq.M(*qubits, key='m')
        )
        return self.sampler.run(circuit, repetitions=repetitions)
        
    def run_experiment(
        self,
        distance: int,
        theta_idx: int,
        loop_idx: int,
        basis: Tuple[int, int],
        theta_range: np.ndarray,
        phi: float,
        repetitions: int,
        use_randomized_compiling: bool = False,
        num_rc_circuits: int = 20,
        folder_name: str = None
    ) -> None:
        """Run a measurement induced entanglement experiment.
        
        Args:
            distance: Distance between probe qubits.
            theta_idx: Index of theta value in theta_range.
            loop_idx: Index of the current loop iteration.
            basis: Tuple of basis states for probe qubits (0=X, 1=Y, 2=Z).
            theta_range: Array of theta values to use.
            phi: Fixed phi value.
            repetitions: Number of times to run the circuit.
            use_randomized_compiling: Whether to use randomized compiling.
            num_rc_circuits: Number of randomized compiled circuits to use.
            folder_name: Name of folder to save results in.
            
        Raises:
            ValueError: If folder_name is not provided.
        """
        if folder_name is None:
            raise ValueError("folder_name must be provided.")
            
        qubits_matrix, probe_qubits, anc_pairs, all_qubits = get_two_qubits_6x6(d=distance)
        circuit = get_circuit(
            qubits_matrix,
            theta=theta_range[theta_idx],
            phi=phi,
            probe_qubits=probe_qubits,
            basis=list(basis),
            anc_pairs=anc_pairs
        )
        circuit.append(cirq.measure(*all_qubits, key='m'))
        
        if use_randomized_compiling:
            circuit_batch = [
                cirq.transformers.gauge_compiling.CZGaugeTransformer(circuit)
                for _ in range(num_rc_circuits)
            ]
            circuit_batch = [
                cirq.merge_single_qubit_moments_to_phxz(circuit)
                for circuit in circuit_batch
            ]
            result = self.sampler.run_batch(
                circuit_batch,
                repetitions=int(repetitions / num_rc_circuits)
            )
            measurements = torch.cat([
                torch.tensor(r[0].measurements['m']).to(int)
                for r in result
            ])
        else:
            result = self.sampler.run(circuit, repetitions=repetitions)
            measurements = torch.tensor(result.measurements['m']).to(int)
            
        # Save measurements
        save_path = f'data/{folder_name}_d={distance}/theta{theta_idx}/loop{loop_idx}/theta={theta_idx}_({basis[0]},{basis[1]}).pt'
        torch.save(measurements, save_path)
        
        print(f'distance={distance}, loop={loop_idx}, size=6x6, theta={theta_idx}, basis={basis[0]},{basis[1]}') 