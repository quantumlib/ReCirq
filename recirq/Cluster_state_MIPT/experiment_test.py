"""Tests for the quantum experiment module.

This module contains tests for the QuantumExperiment class and its methods.
"""

import unittest
from unittest.mock import MagicMock, patch
import cirq
import cirq_google as cg
import numpy as np
import torch
from recirq.measurement_entanglement.experiment import QuantumExperiment


class QuantumExperimentTest(unittest.TestCase):
    """Test cases for the QuantumExperiment class."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_id = 'test-project'
        self.processor_id = 'test-processor'
        self.device_config_name = 'test-config'
        self.snapshot_id = 'test-snapshot'
        self.run_name = 'test-run'

    @patch('cirq_google.get_engine')
    def test_init_with_snapshot(self, mock_get_engine):
        """Test initialization with snapshot_id."""
        mock_engine = MagicMock()
        mock_processor = MagicMock()
        mock_sampler = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.get_processor.return_value = mock_processor
        mock_processor.get_sampler.return_value = mock_sampler

        experiment = QuantumExperiment(
            project_id=self.project_id,
            processor_id=self.processor_id,
            device_config_name=self.device_config_name,
            snapshot_id=self.snapshot_id
        )

        mock_get_engine.assert_called_once_with(self.project_id)
        mock_engine.get_processor.assert_called_once_with(self.processor_id)
        mock_processor.get_sampler.assert_called_once_with(
            device_config_name=self.device_config_name,
            snapshot_id=self.snapshot_id
        )
        self.assertIsNotNone(experiment.simulator)

    @patch('cirq_google.get_engine')
    def test_init_with_run_name(self, mock_get_engine):
        """Test initialization with run_name."""
        mock_engine = MagicMock()
        mock_sampler = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.get_sampler.return_value = mock_sampler

        experiment = QuantumExperiment(
            project_id=self.project_id,
            processor_id=self.processor_id,
            device_config_name=self.device_config_name,
            run_name=self.run_name
        )

        mock_get_engine.assert_called_once_with(self.project_id)
        mock_engine.get_sampler.assert_called_once_with(
            processor_id=self.processor_id,
            run_name=self.run_name,
            device_config_name=self.device_config_name
        )
        self.assertIsNotNone(experiment.simulator)

    def test_init_without_snapshot_or_run_name(self):
        """Test initialization without snapshot_id or run_name."""
        with self.assertRaises(ValueError):
            QuantumExperiment(
                project_id=self.project_id,
                processor_id=self.processor_id,
                device_config_name=self.device_config_name
            )

    @patch('cirq_google.get_engine')
    def test_test_bell_state(self, mock_get_engine):
        """Test Bell state experiment."""
        mock_engine = MagicMock()
        mock_processor = MagicMock()
        mock_sampler = MagicMock()
        mock_result = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.get_processor.return_value = mock_processor
        mock_processor.get_sampler.return_value = mock_sampler
        mock_sampler.run.return_value = mock_result

        experiment = QuantumExperiment(
            project_id=self.project_id,
            processor_id=self.processor_id,
            device_config_name=self.device_config_name,
            snapshot_id=self.snapshot_id
        )

        result = experiment.test_bell_state(repetitions=1000)
        self.assertEqual(result, mock_result)
        mock_sampler.run.assert_called_once()

    @patch('cirq_google.get_engine')
    def test_run_experiment(self, mock_get_engine):
        """Test running a measurement induced entanglement experiment."""
        mock_engine = MagicMock()
        mock_processor = MagicMock()
        mock_sampler = MagicMock()
        mock_result = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.get_processor.return_value = mock_processor
        mock_processor.get_sampler.return_value = mock_sampler
        mock_sampler.run.return_value = mock_result

        experiment = QuantumExperiment(
            project_id=self.project_id,
            processor_id=self.processor_id,
            device_config_name=self.device_config_name,
            snapshot_id=self.snapshot_id
        )

        # Test parameters
        distance = 6
        theta_idx = 0
        loop_idx = 0
        basis = (0, 0)
        theta_range = np.linspace(0, np.pi, 11)
        phi = 0.0
        repetitions = 1000
        folder_name = 'test_folder'

        # Test without randomized compiling
        experiment.run_experiment(
            distance=distance,
            theta_idx=theta_idx,
            loop_idx=loop_idx,
            basis=basis,
            theta_range=theta_range,
            phi=phi,
            repetitions=repetitions,
            folder_name=folder_name
        )

        mock_sampler.run.assert_called_once()

        # Test with randomized compiling
        experiment.run_experiment(
            distance=distance,
            theta_idx=theta_idx,
            loop_idx=loop_idx,
            basis=basis,
            theta_range=theta_range,
            phi=phi,
            repetitions=repetitions,
            use_randomized_compiling=True,
            num_rc_circuits=20,
            folder_name=folder_name
        )

        mock_sampler.run_batch.assert_called_once()

    def test_run_experiment_without_folder_name(self):
        """Test running experiment without folder_name."""
        experiment = QuantumExperiment(
            project_id=self.project_id,
            processor_id=self.processor_id,
            device_config_name=self.device_config_name,
            snapshot_id=self.snapshot_id
        )

        with self.assertRaises(ValueError):
            experiment.run_experiment(
                distance=6,
                theta_idx=0,
                loop_idx=0,
                basis=(0, 0),
                theta_range=np.linspace(0, np.pi, 11),
                phi=0.0,
                repetitions=1000
            )


if __name__ == '__main__':
    unittest.main() 