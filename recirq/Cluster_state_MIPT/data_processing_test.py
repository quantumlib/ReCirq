"""Tests for the data processing module."""

import unittest
import torch
import os
import tempfile
from recirq.Cluster_state_MIPT import data_processing
from recirq.Cluster_state_MIPT import config


class DataProcessingTest(unittest.TestCase):
    """Test cases for data processing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = config.get_config()
        self.dtype = self.config['processing']['dtype']
        self.device = self.config['processing']['device']
        
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.config['paths']['data_dir'] = self.test_dir

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_process_measurements(self):
        """Test processing of measurement data."""
        # Create sample measurement data
        num_reps = 100
        num_qubits = 4
        measurements = torch.randint(0, 2, (num_reps, num_qubits), 
                                   dtype=torch.int64, device=self.device)
        
        # Process measurements
        prep_seq, shadow_state, rhoS = data_processing.process_measurements(
            measurements, self.dtype, self.device)
        
        # Check output shapes and types
        self.assertEqual(prep_seq.shape, (num_reps, num_qubits))
        self.assertEqual(shadow_state.shape, (num_reps, num_qubits))
        self.assertEqual(rhoS.shape, (num_reps, 2**num_qubits, 2**num_qubits))
        
        self.assertEqual(prep_seq.dtype, torch.int64)
        self.assertEqual(shadow_state.dtype, torch.int64)
        self.assertEqual(rhoS.dtype, self.dtype)

    def test_shuffle_data(self):
        """Test data shuffling functionality."""
        # Create sample data
        num_reps = 100
        num_qubits = 4
        prep_seq = torch.randint(0, 2, (num_reps, num_qubits), 
                               dtype=torch.int64, device=self.device)
        shadow_state = torch.randint(0, 2, (num_reps, num_qubits), 
                                   dtype=torch.int64, device=self.device)
        rhoS = torch.randn(num_reps, 2**num_qubits, 2**num_qubits, 
                          dtype=self.dtype, device=self.device)
        
        # Shuffle data
        shuffled_prep_seq, shuffled_shadow_state, shuffled_rhoS = data_processing.shuffle_data(
            prep_seq, shadow_state, rhoS, seed=42)
        
        # Check that data is shuffled but preserved
        self.assertEqual(shuffled_prep_seq.shape, prep_seq.shape)
        self.assertEqual(shuffled_shadow_state.shape, shadow_state.shape)
        self.assertEqual(shuffled_rhoS.shape, rhoS.shape)
        
        # Check that all elements are preserved
        self.assertEqual(torch.sort(prep_seq.flatten())[0], 
                        torch.sort(shuffled_prep_seq.flatten())[0])
        self.assertEqual(torch.sort(shadow_state.flatten())[0], 
                        torch.sort(shuffled_shadow_state.flatten())[0])

    def test_process_all_data(self):
        """Test processing of all experiment data."""
        # Create sample data for one experiment
        folder_name = 'test'
        distance = 3
        theta_idx = 0
        loop_idx = 0
        
        # Create directory structure
        exp_dir = os.path.join(self.test_dir, f'{folder_name}_d={distance}', 
                              f'theta{theta_idx}', f'loop{loop_idx}')
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create and save sample measurement data
        num_reps = 100
        num_qubits = 4
        measurements = torch.randint(0, 2, (num_reps, num_qubits), 
                                   dtype=torch.int64, device=self.device)
        torch.save(measurements, os.path.join(exp_dir, f'theta={theta_idx}_({0},{0}).pt'))
        
        # Process all data
        data_processing.process_all_data(
            folder_name=folder_name,
            distances=[distance],
            theta_indices=[theta_idx],
            num_loops=1,
            seed=42)
        
        # Check that processed data files exist
        output_dir = os.path.join(self.test_dir, 'processed')
        self.assertTrue(os.path.exists(os.path.join(output_dir, 
            f'all_prepseq_theta={theta_idx}.pt')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 
            f'all_shadow_state_theta={theta_idx}.pt')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 
            f'all_rhoS_theta={theta_idx}.pt')))


if __name__ == '__main__':
    unittest.main() 