"""Tests for the quantum information theory utility functions."""

import unittest
import torch
import numpy as np
from recirq.measurement_entanglement import utils


# Test constants
DTYPE = torch.complex128
DEVICE = torch.device("cpu")
EPSILON = 1e-5
RNG = np.random.default_rng()


class UtilsTest(unittest.TestCase):
    """Test cases for quantum information theory utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.dtype = DTYPE
        self.device = DEVICE
        self.basis = utils.setup_pauli_matrices(self.device, self.dtype)

    def test_setup_pauli_matrices(self):
        """Test Pauli matrices setup and properties."""
        self.assertIsNotNone(utils.PAULI_MATRICES)
        self.assertEqual(utils.PAULI_MATRICES.shape, (4, 2, 2))
        self.assertEqual(self.basis.shape, (3, 2, 2))
        
        # Test Pauli matrix properties
        for i in range(4):
            pauli = utils.PAULI_MATRICES[i]
            # Hermitian
            self.assertTrue(torch.allclose(pauli.conj().T, pauli))
            # Unitary
            self.assertTrue(torch.allclose(pauli @ pauli.conj().T, torch.eye(2, dtype=self.dtype, device=self.device)))
            # Trace zero (except identity)
            if i > 0:
                self.assertTrue(torch.allclose(torch.trace(pauli), torch.tensor(0.0, dtype=self.dtype)))

    def test_eps(self):
        """Test noise addition to density matrix."""
        # Test pure state
        rho = torch.tensor([[1, 0], [0, 0]], dtype=self.dtype, device=self.device)[None, ...]
        noisy_rho = utils.eps(rho, e=0.1)
        self.assertEqual(noisy_rho.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(torch.trace(noisy_rho[0]), torch.tensor(1.0, dtype=self.dtype)))
        self.assertTrue(torch.allclose(noisy_rho[0].conj().T, noisy_rho[0]))  # Hermitian
        
        # Test maximally mixed state
        rho = torch.eye(2, dtype=self.dtype, device=self.device)[None, ...] / 2
        noisy_rho = utils.eps(rho, e=0.1)
        self.assertTrue(torch.allclose(noisy_rho, rho))  # Should be unchanged
        
        # Test error cases
        with self.assertRaises(ValueError):
            utils.eps(rho, e=1.5)  # e > 1
        with self.assertRaises(ValueError):
            utils.eps(rho, e=-0.1)  # e < 0
        with self.assertRaises(ValueError):
            invalid_rho = torch.tensor([[2, 0], [0, 0]], dtype=self.dtype, device=self.device)[None, ...]
            utils.eps(invalid_rho)  # Trace > 1

    def test_blogm(self):
        """Test matrix logarithm calculation."""
        # Test positive definite matrix
        A = torch.tensor([[2, 1], [1, 2]], dtype=self.dtype, device=self.device)[None, ...]
        logA = utils.blogm(A)
        self.assertEqual(logA.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(logA[0].conj().T, logA[0]))  # Hermitian
        
        # Test identity matrix
        I = torch.eye(2, dtype=self.dtype, device=self.device)[None, ...]
        logI = utils.blogm(I)
        self.assertTrue(torch.allclose(logI, torch.zeros_like(I)))  # log(1) = 0
        
        # Test error cases
        with self.assertRaises(ValueError):
            invalid_A = torch.tensor([[1, 2], [2, 1]], dtype=self.dtype, device=self.device)[None, ...]
            utils.blogm(invalid_A)  # Not a density matrix

    def test_bSqc(self):
        """Test squashed concurrence calculation."""
        # Test pure states
        rhoQ = torch.tensor([[1, 0], [0, 0]], dtype=self.dtype, device=self.device)[None, ...]
        rhoC = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=self.dtype, device=self.device)[None, ...]
        sqc = utils.bSqc(rhoQ, rhoC)
        self.assertEqual(sqc.shape, (1,))
        self.assertTrue(torch.all(sqc <= 0))  # Should be non-positive
        
        # Test maximally mixed states
        rhoQ = torch.eye(2, dtype=self.dtype, device=self.device)[None, ...] / 2
        rhoC = torch.eye(2, dtype=self.dtype, device=self.device)[None, ...] / 2
        sqc = utils.bSqc(rhoQ, rhoC)
        self.assertTrue(torch.allclose(sqc, torch.tensor([0.0], dtype=self.dtype, device=self.device)))
        
        # Test error cases
        with self.assertRaises(ValueError):
            rhoQ_cuda = rhoQ.to('cuda') if torch.cuda.is_available() else rhoQ
            utils.bSqc(rhoQ_cuda, rhoC)  # Device mismatch

    def test_Neg(self):
        """Test negativity calculation."""
        # Test separable states
        rhoS = torch.eye(4, dtype=self.dtype, device=self.device)[None, ...] / 4
        rhoC = torch.eye(4, dtype=self.dtype, device=self.device)[None, ...] / 4
        neg = utils.Neg(rhoS, rhoC)
        self.assertEqual(neg.shape, (1,))
        self.assertTrue(torch.all(neg >= 0))  # Should be non-negative
        self.assertTrue(torch.allclose(neg, torch.tensor([0.0], dtype=self.dtype, device=self.device)))
        
        # Test Bell state
        bell = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], 
                          dtype=self.dtype, device=self.device)[None, ...] / 2
        neg = utils.Neg(bell, bell)
        self.assertTrue(torch.all(neg > 0))  # Should be positive for entangled state
        
        # Test error cases
        with self.assertRaises(ValueError):
            invalid_rho = torch.eye(2, dtype=self.dtype, device=self.device)[None, ...]
            utils.Neg(invalid_rho, invalid_rho)  # Wrong size

    def test_Sa(self):
        """Test von Neumann entropy calculation."""
        # Test pure states
        rhoS = torch.eye(4, dtype=self.dtype, device=self.device)[None, ...] / 4
        rhoC = torch.eye(4, dtype=self.dtype, device=self.device)[None, ...] / 4
        entropy = utils.Sa(rhoS, rhoC)
        self.assertEqual(entropy.shape, (1,))
        self.assertTrue(torch.all(entropy <= 0))  # Should be non-positive
        
        # Test maximally mixed states
        rhoS = torch.eye(4, dtype=self.dtype, device=self.device)[None, ...] / 4
        rhoC = torch.eye(4, dtype=self.dtype, device=self.device)[None, ...] / 4
        entropy = utils.Sa(rhoS, rhoC)
        self.assertTrue(torch.allclose(entropy, torch.tensor([0.0], dtype=self.dtype, device=self.device)))
        
        # Test error cases
        with self.assertRaises(ValueError):
            invalid_rho = torch.eye(2, dtype=self.dtype, device=self.device)[None, ...]
            utils.Sa(invalid_rho, invalid_rho)  # Wrong size


if __name__ == '__main__':
    unittest.main() 