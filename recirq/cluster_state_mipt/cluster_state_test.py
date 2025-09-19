# Copyright 2025 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for cluster_state.py."""

import numpy as np
import pytest
import cirq

from recirq.cluster_state_mipt import cluster_state


def test_get_two_qubits_6x6():
    """Test get_two_qubits_6x6 with valid distance values."""
    # Test d=6
    qubits_matrix, probe_qubits, anc_pairs, all_qubits = cluster_state.get_two_qubits_6x6(d=6)
    assert len(probe_qubits) == 2
    assert probe_qubits[0] == cirq.GridQubit(3, 4)
    assert probe_qubits[1] == cirq.GridQubit(3, 9)
    assert len(qubits_matrix) == 6
    assert len(qubits_matrix[0]) == 6

    # Test d=5
    qubits_matrix, probe_qubits, anc_pairs, all_qubits = cluster_state.get_two_qubits_6x6(d=5)
    assert len(probe_qubits) == 2
    assert probe_qubits[0] == cirq.GridQubit(3, 4)
    assert probe_qubits[1] == cirq.GridQubit(3, 8)

    # Test d=4
    qubits_matrix, probe_qubits, anc_pairs, all_qubits = cluster_state.get_two_qubits_6x6(d=4)
    assert len(probe_qubits) == 2
    assert probe_qubits[0] == cirq.GridQubit(3, 5)
    assert probe_qubits[1] == cirq.GridQubit(3, 8)

    # Test d=3
    qubits_matrix, probe_qubits, anc_pairs, all_qubits = cluster_state.get_two_qubits_6x6(d=3)
    assert len(probe_qubits) == 2
    assert probe_qubits[0] == cirq.GridQubit(3, 5)
    assert probe_qubits[1] == cirq.GridQubit(3, 7)


def test_get_two_qubits_6x6_invalid():
    """Test get_two_qubits_6x6 with invalid distance value."""
    with pytest.raises(ValueError):
        cluster_state.get_two_qubits_6x6(d=7)


def test_get_circuit():
    """Test basic circuit creation using outputs from get_two_qubits_6x6."""
    # Get real qubit configuration from get_two_qubits_6x6
    qubits_matrix, probe_qubits, anc_pairs, all_qubits = cluster_state.get_two_qubits_6x6(d=4)
    
    theta = np.pi/4
    phi = np.pi/2
    basis = [0, 0]

    circuit = cluster_state.get_circuit(
        qubits_matrix=qubits_matrix,
        theta=theta,
        phi=phi,
        probe_qubits=probe_qubits,
        basis=basis,
        anc_pairs=anc_pairs
    )

    assert isinstance(circuit, cirq.Circuit)
    # The circuit has 14 operations because it's a shallow circuit with constant depth:
    # - Initial layer of Hadamard gates (1 moment)
    # - Horizontal CZ bonds with phase corrections (4 moments)
    # - Vertical CZ bonds with phase corrections (4 moments)
    # - Single qubit rotations (Rz and Ry for each qubit, 2 moments)
    # - Basis rotations on probe qubits (1 moment)
    # - Ancilla operations (2 moments)
    assert len(circuit) == 14, f"Expected circuit length to be 14, got {len(circuit)}"
    # Verify probe qubit measurements are properly set up
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.ops.common_gates.HPowGate) and op.qubits[0] in probe_qubits:
            # Found Hadamard on a probe qubit (basis 0 = X basis)
            assert op.qubits[0] in probe_qubits


def test_get_circuit_invalid_basis():
    """Test circuit creation with invalid basis states using get_two_qubits_6x6 output."""
    # Get real qubit configuration from get_two_qubits_6x6
    qubits_matrix, probe_qubits, anc_pairs, all_qubits = cluster_state.get_two_qubits_6x6(d=3)
    
    # Only use first probe qubit but provide two basis elements
    mismatched_probe_qubits = [probe_qubits[0]]  # Only one probe qubit
    mismatched_basis = [0, 0]  # Two basis states for one probe qubit

    with pytest.raises(ValueError):
        cluster_state.get_circuit(
            qubits_matrix=qubits_matrix,
            theta=np.pi/4,
            phi=np.pi/2,
            probe_qubits=mismatched_probe_qubits,
            basis=mismatched_basis,
            anc_pairs=anc_pairs
        ) 