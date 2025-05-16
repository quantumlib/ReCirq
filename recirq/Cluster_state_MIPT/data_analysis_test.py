"""Tests for the data analysis module."""

import unittest
import torch
from recirq.measurement_entanglement.data_analysis import (
    eps,
    cross,
    side,
    corner,
    contract,
    analyze_data
)


class DataAnalysisTest(unittest.TestCase):
    """Test cases for data analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.dtype = torch.complex128
        self.device = torch.device("cpu")
        self.theta = torch.tensor(0.0, dtype=self.dtype)
        self.phi = torch.tensor(0.0, dtype=self.dtype)

    def test_eps(self):
        """Test the eps function."""
        rho = torch.eye(2, dtype=self.dtype, device=self.device)
        noisy_rho = eps(rho, e=0.1)
        self.assertEqual(noisy_rho.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(noisy_rho[0], rho))

    def test_cross(self):
        """Test the cross function."""
        kernel = torch.eye(2, dtype=self.dtype, device=self.device)
        result = cross(0, self.theta, self.phi, kernel)
        self.assertIsInstance(result, torch.Tensor)

    def test_side(self):
        """Test the side function."""
        kernel = torch.eye(2, dtype=self.dtype, device=self.device)
        result = side(0, self.theta, self.phi, kernel)
        self.assertIsInstance(result, torch.Tensor)

    def test_corner(self):
        """Test the corner function."""
        kernel = torch.eye(2, dtype=self.dtype, device=self.device)
        result = corner(0, self.theta, self.phi, kernel)
        self.assertIsInstance(result, torch.Tensor)

    def test_contract(self):
        """Test the contract function."""
        src = torch.ones((2, 2), dtype=self.dtype, device=self.device)
        dst = torch.ones((2, 2), dtype=self.dtype, device=self.device)
        src_idx = [0, 1]
        dst_idx = [0, 1]
        result, new_idx = contract(src, src_idx, dst, dst_idx)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(new_idx, list)


if __name__ == '__main__':
    unittest.main() 