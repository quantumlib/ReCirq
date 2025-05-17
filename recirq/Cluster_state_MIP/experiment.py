"""Implementation of cluster state experiments with simulator support.

This module provides functionality for creating and running cluster state experiments
using either quantum hardware or simulators.
"""

import cirq
import cirq_google as cg
import numpy as np
from typing import List, Optional, Tuple
from recirq.Cluster_state_MIPT.cluster_state import get_two_qubits_6x6, get_circuit


class QuantumExperiment:
    """A class for running cluster state experiments on quantum hardware or simulators."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        processor_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        run_name: Optional[str] = None,
        use_simulator: bool = True
    ):
        """Initialize the experiment.

        Args:
            project_id: Google Cloud project ID (required for hardware)
            processor_id: Quantum processor ID (required for hardware)
            snapshot_id: Calibration snapshot ID (optional)
            run_name: Run name for the experiment (optional)
            use_simulator: Whether to use simulator instead of hardware
        """
        self.use_simulator = use_simulator
        
        if use_simulator:
            self.sampler = cirq.Simulator()
        else:
            if not project_id or not processor_id:
                raise ValueError("project_id and processor_id are required for hardware experiments")
            
            engine = cg.get_engine(project_id)
            if snapshot_id:
                processor = engine.get_processor(processor_id)
                self.sampler = processor.get_sampler(snapshot_id=snapshot_id)
            elif run_name:
                self.sampler = engine.get_sampler(processor_id=processor_id, run_name=run_name)
            else:
                raise ValueError("Either snapshot_id or run_name must be provided for hardware experiments")

    def test_bell_state(self, repetitions: int = 1000) -> cirq.Result:
        """Run a simple Bell state test.

        Args:
            repetitions: Number of times to run the circuit

        Returns:
            Measurement results
        """
        # Create a simple Bell state circuit
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='m')
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
        folder_name: str,
        use_randomized_compiling: bool = False,
        num_rc_circuits: int = 20
    ) -> cirq.Result:
        """Run a cluster state experiment.

        Args:
            distance: Size of the grid (3, 4, 5, or 6)
            theta_idx: Index into theta_range
            loop_idx: Loop index for multiple runs
            basis: Measurement basis for probe qubits (0=X, 1=Y, 2=Z)
            theta_range: Range of theta values to sweep
            phi: Fixed phi value
            repetitions: Number of times to run the circuit
            folder_name: Name of folder to save results
            use_randomized_compiling: Whether to use randomized compiling
            num_rc_circuits: Number of randomized compiling circuits

        Returns:
            Measurement results
        """
        # Get qubit layout
        qubits_matrix, probe_qubits, anc_pairs, _ = get_two_qubits_6x6(d=distance)
        
        # Create circuit
        circuit = get_circuit(
            qubits_matrix=qubits_matrix,
            theta=theta_range[theta_idx],
            phi=phi,
            probe_qubits=probe_qubits,
            basis=list(basis),
            anc_pairs=anc_pairs
        )
        
        # Add measurements
        circuit.append(cirq.measure(*probe_qubits, key='m'))
        
        if use_randomized_compiling:
            # Create multiple circuits with randomized compiling
            circuits = []
            for _ in range(num_rc_circuits):
                rc_circuit = circuit.copy()
                # Add random single-qubit gates
                for q in circuit.all_qubits():
                    if np.random.random() < 0.5:
                        rc_circuit.append(cirq.X(q))
                circuits.append(rc_circuit)
            
            # Run all circuits
            results = []
            for rc_circuit in circuits:
                result = self.sampler.run(rc_circuit, repetitions=repetitions)
                results.append(result)
            return results
        else:
            # Run single circuit
            return self.sampler.run(circuit, repetitions=repetitions) 