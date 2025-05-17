"""Tests for the quantum experiment module."""

import unittest
import cirq
import numpy as np
from recirq.Cluster_state_MIPT.experiment import QuantumExperiment


class QuantumExperimentTest(unittest.TestCase):
    """Test cases for the QuantumExperiment class."""

    def setUp(self):
        """Set up test fixtures."""
        self.experiment = QuantumExperiment(use_simulator=True)

    def test_bell_state(self):
        """Test Bell state experiment."""
        result = self.experiment.test_bell_state(repetitions=1000)
        
        # Check that we got the expected number of measurements
        self.assertEqual(len(result.measurements['m']), 1000)
        
        # Check that measurements are either [0,0] or [1,1] (Bell state)
        measurements = result.measurements['m']
        self.assertTrue(all(m[0] == m[1] for m in measurements))
        
        # Check that we have roughly equal number of [0,0] and [1,1]
        zeros = sum(1 for m in measurements if m[0] == 0)
        ones = sum(1 for m in measurements if m[0] == 1)
        self.assertAlmostEqual(zeros / len(measurements), 0.5, delta=0.1)
        self.assertAlmostEqual(ones / len(measurements), 0.5, delta=0.1)

    def test_run_experiment(self):
        """Test running a cluster state experiment."""
        # Test parameters
        distance = 3  # Use smallest grid for testing
        theta_idx = 0
        loop_idx = 0
        basis = (0, 0)  # X basis
        theta_range = np.linspace(0, np.pi, 11)
        phi = 0.0
        repetitions = 100
        folder_name = 'test_folder'

        # Test without randomized compiling
        result = self.experiment.run_experiment(
            distance=distance,
            theta_idx=theta_idx,
            loop_idx=loop_idx,
            basis=basis,
            theta_range=theta_range,
            phi=phi,
            repetitions=repetitions,
            folder_name=folder_name
        )
        
        # Check that we got the expected number of measurements
        self.assertEqual(len(result.measurements['m']), repetitions)
        
        # Test with randomized compiling
        results = self.experiment.run_experiment(
            distance=distance,
            theta_idx=theta_idx,
            loop_idx=loop_idx,
            basis=basis,
            theta_range=theta_range,
            phi=phi,
            repetitions=repetitions,
            folder_name=folder_name,
            use_randomized_compiling=True,
            num_rc_circuits=5
        )
        
        # Check that we got results for all randomized compiling circuits
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertEqual(len(result.measurements['m']), repetitions)

    def test_invalid_distance(self):
        """Test running experiment with invalid distance."""
        with self.assertRaises(ValueError):
            self.experiment.run_experiment(
                distance=2,  # Invalid distance
                theta_idx=0,
                loop_idx=0,
                basis=(0, 0),
                theta_range=np.linspace(0, np.pi, 11),
                phi=0.0,
                repetitions=100,
                folder_name='test_folder'
            )

    def test_invalid_basis(self):
        """Test running experiment with invalid basis."""
        with self.assertRaises(ValueError):
            self.experiment.run_experiment(
                distance=3,
                theta_idx=0,
                loop_idx=0,
                basis=(0,),  # Invalid basis (only one value for two qubits)
                theta_range=np.linspace(0, np.pi, 11),
                phi=0.0,
                repetitions=100,
                folder_name='test_folder'
            )


if __name__ == '__main__':
    unittest.main() 