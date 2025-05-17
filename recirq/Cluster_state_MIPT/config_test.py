"""Tests for the configuration module."""

import unittest
import torch
from recirq.Cluster_state_MIPT import config


class ConfigTest(unittest.TestCase):
    """Test cases for configuration parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = config.get_config()

    def test_device_config(self):
        """Test device configuration parameters."""
        device_config = self.config['device']
        self.assertIsInstance(device_config, dict)
        self.assertIn('project_id', device_config)
        self.assertIn('processor_id', device_config)
        self.assertIn('device_config_name', device_config)
        self.assertIn('snapshot_id', device_config)
        self.assertIn('run_name', device_config)

    def test_experiment_config(self):
        """Test experiment configuration parameters."""
        exp_config = self.config['experiment']
        self.assertIsInstance(exp_config, dict)
        self.assertIn('folder_name', exp_config)
        self.assertIn('distances', exp_config)
        self.assertIn('theta_indices', exp_config)
        self.assertIn('num_loops', exp_config)
        self.assertIn('seed', exp_config)
        
        self.assertIsInstance(exp_config['distances'], list)
        self.assertIsInstance(exp_config['theta_indices'], list)
        self.assertIsInstance(exp_config['num_loops'], int)
        self.assertIsInstance(exp_config['seed'], int)

    def test_processing_config(self):
        """Test data processing configuration parameters."""
        proc_config = self.config['processing']
        self.assertIsInstance(proc_config, dict)
        self.assertIn('dtype', proc_config)
        self.assertIn('device', proc_config)
        self.assertIn('noise_parameter', proc_config)
        
        self.assertEqual(proc_config['dtype'], torch.complex128)
        self.assertIsInstance(proc_config['device'], torch.device)
        self.assertIsInstance(proc_config['noise_parameter'], float)

    def test_analysis_config(self):
        """Test analysis configuration parameters."""
        analysis_config = self.config['analysis']
        self.assertIsInstance(analysis_config, dict)
        self.assertIn('phi', analysis_config)
        self.assertIn('theta_range', analysis_config)
        
        self.assertIsInstance(analysis_config['phi'], torch.Tensor)
        self.assertIsInstance(analysis_config['theta_range'], torch.Tensor)
        self.assertEqual(len(analysis_config['theta_range']), 11)

    def test_path_config(self):
        """Test path configuration parameters."""
        path_config = self.config['paths']
        self.assertIsInstance(path_config, dict)
        self.assertIn('data_dir', path_config)
        self.assertIn('results_dir', path_config)
        self.assertIn('plots_dir', path_config)
        
        self.assertIsInstance(path_config['data_dir'], str)
        self.assertIsInstance(path_config['results_dir'], str)
        self.assertIsInstance(path_config['plots_dir'], str)


if __name__ == '__main__':
    unittest.main() 