"""Tests for the cluster state module.

This module contains tests for the cluster state circuit generation functions.
"""

import unittest
import cirq
import numpy as np
from recirq.measurement_entanglement.cluster_state import get_two_qubits_6x6, get_circuit


class ClusterStateTest(unittest.TestCase):
    """Test cases for cluster state functions."""

    def test_get_two_qubits_6x6_valid_distances(self):
        """Test get_two_qubits_6x6 with valid distances."""
        valid_distances = [3, 4, 5, 6]
        for d in valid_distances:
            qubits_matrix, probe_qubits, anc_pairs, all_qubits = get_two_qubits_6x6(d=d)
            
            # Check qubits_matrix structure
            self.assertEqual(len(qubits_matrix), 6)
            self.assertEqual(len(qubits_matrix[0]), 6)
            self.assertTrue(all(isinstance(q, cirq.GridQubit) for row in qubits_matrix for q in row))
            
            # Check probe qubits
            self.assertEqual(len(probe_qubits), 2)
            self.assertTrue(all(isinstance(q, cirq.GridQubit) for q in probe_qubits))
            
            # Check ancilla pairs
            self.assertEqual(len(anc_pairs), 1)
            self.assertTrue(all(isinstance(pair[0], cirq.GridQubit) and isinstance(pair[1], cirq.GridQubit)
                              for pair_set in anc_pairs for pair in pair_set))
            
            # Check all_qubits
            self.assertTrue(isinstance(all_qubits, np.ndarray))
            self.assertTrue(all(isinstance(q, cirq.GridQubit) for q in all_qubits))

    def test_get_two_qubits_6x6_invalid_distance(self):
        """Test get_two_qubits_6x6 with invalid distance."""
        with self.assertRaises(ValueError):
            get_two_qubits_6x6(d=2)

    def test_get_two_qubits_6x6_probe_qubit_positions(self):
        """Test probe qubit positions for different distances."""
        test_cases = [
            (6, [(3, 4), (3, 9)]),
            (5, [(3, 4), (3, 8)]),
            (4, [(3, 5), (3, 8)]),
            (3, [(3, 5), (3, 7)])
        ]
        
        for d, expected_positions in test_cases:
            _, probe_qubits, _, _ = get_two_qubits_6x6(d=d)
            actual_positions = [(q.row, q.col) for q in probe_qubits]
            self.assertEqual(actual_positions, expected_positions)

    def test_get_circuit_basic(self):
        """Test basic circuit generation."""
        qubits_matrix, probe_qubits, anc_pairs, _ = get_two_qubits_6x6(d=6)
        theta = 0.0
        phi = 0.0
        basis = [0, 0]
        
        circuit = get_circuit(
            qubits_matrix=qubits_matrix,
            theta=theta,
            phi=phi,
            probe_qubits=probe_qubits,
            basis=basis,
            anc_pairs=anc_pairs
        )
        
        self.assertIsInstance(circuit, cirq.Circuit)
        self.assertTrue(len(circuit) > 0)

    def test_get_circuit_basis_mismatch(self):
        """Test circuit generation with mismatched basis states."""
        qubits_matrix, probe_qubits, anc_pairs, _ = get_two_qubits_6x6(d=6)
        theta = 0.0
        phi = 0.0
        basis = [0]  # Only one basis state for two probe qubits
        
        with self.assertRaises(ValueError):
            get_circuit(
                qubits_matrix=qubits_matrix,
                theta=theta,
                phi=phi,
                probe_qubits=probe_qubits,
                basis=basis,
                anc_pairs=anc_pairs
            )

    def test_get_circuit_basis_rotations(self):
        """Test basis rotations in circuit generation."""
        qubits_matrix, probe_qubits, anc_pairs, _ = get_two_qubits_6x6(d=6)
        theta = 0.0
        phi = 0.0
        
        # Test X basis (0)
        circuit_x = get_circuit(
            qubits_matrix=qubits_matrix,
            theta=theta,
            phi=phi,
            probe_qubits=probe_qubits,
            basis=[0, 0],
            anc_pairs=anc_pairs
        )
        self.assertTrue(any(isinstance(op.gate, cirq.HGate) for op in circuit_x.all_operations()))
        
        # Test Y basis (1)
        circuit_y = get_circuit(
            qubits_matrix=qubits_matrix,
            theta=theta,
            phi=phi,
            probe_qubits=probe_qubits,
            basis=[1, 1],
            anc_pairs=anc_pairs
        )
        self.assertTrue(any(isinstance(op.gate, cirq.Rx) for op in circuit_y.all_operations()))
        
        # Test Z basis (2)
        circuit_z = get_circuit(
            qubits_matrix=qubits_matrix,
            theta=theta,
            phi=phi,
            probe_qubits=probe_qubits,
            basis=[2, 2],
            anc_pairs=anc_pairs
        )
        # Z basis should not have any special rotations
        self.assertFalse(any(isinstance(op.gate, (cirq.HGate, cirq.Rx)) for op in circuit_z.all_operations()))

    def test_get_circuit_ancilla_operations(self):
        """Test ancilla qubit operations in circuit generation."""
        qubits_matrix, probe_qubits, anc_pairs, _ = get_two_qubits_6x6(d=6)
        theta = 0.0
        phi = 0.0
        basis = [0, 0]
        
        circuit = get_circuit(
            qubits_matrix=qubits_matrix,
            theta=theta,
            phi=phi,
            probe_qubits=probe_qubits,
            basis=basis,
            anc_pairs=anc_pairs
        )
        
        # Check for Hadamard gates on ancilla qubits
        ancilla_qubits = [pair[1] for pair_set in anc_pairs for pair in pair_set]
        self.assertTrue(any(isinstance(op.gate, cirq.HGate) and op.qubits[0] in ancilla_qubits
                          for op in circuit.all_operations()))
        
        # Check for CZ gates between physical and ancilla qubits
        self.assertTrue(any(isinstance(op.gate, cirq.CZPowGate) and
                          any(q in ancilla_qubits for q in op.qubits)
                          for op in circuit.all_operations()))


if __name__ == '__main__':
    unittest.main() 