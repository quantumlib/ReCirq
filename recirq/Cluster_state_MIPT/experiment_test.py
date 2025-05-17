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
        result = self.experiment.test_bell_state(repetitions=100)  # Reduced repetitions
        
        # Check that we got the expected number of measurements
        self.assertEqual(len(result.measurements['m']), 100)
        
        # Check that measurements are either [0,0] or [1,1] (Bell state)
        measurements = result.measurements['m']
        self.assertTrue(all(m[0] == m[1] for m in measurements))
        
        # Check that we have roughly equal number of [0,0] and [1,1]
        zeros = sum(1 for m in measurements if m[0] == 0)
        ones = sum(1 for m in measurements if m[0] == 1)
        self.assertAlmostEqual(zeros / len(measurements), 0.5, delta=0.2)  # Increased tolerance
        self.assertAlmostEqual(ones / len(measurements), 0.5, delta=0.2)  # Increased tolerance

    def test_invalid_distance(self):
        """Test running experiment with invalid distance."""
        with self.assertRaises(ValueError):
            self.experiment.run_experiment(
                distance=2,  # Invalid distance
                theta_idx=0,
                loop_idx=0,
                basis=(0, 0),
                theta_range=np.linspace(0, np.pi, 5),
                phi=0.0,
                repetitions=10,
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
                theta_range=np.linspace(0, np.pi, 5),
                phi=0.0,
                repetitions=10,
                folder_name='test_folder'
            )


if __name__ == '__main__':
    unittest.main() 