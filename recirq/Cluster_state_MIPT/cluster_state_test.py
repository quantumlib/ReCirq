"""Tests for cluster_state.py."""

import numpy as np
import pytest
import cirq

from recirq.Cluster_state_MIPT import cluster_state


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
    """Test basic circuit creation."""
    qubits_matrix = [
        [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)],
        [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]
    ]
    probe_qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)]
    theta = np.pi/4
    phi = np.pi/2
    basis = [0, 0]
    anc_pairs = []

    circuit = cluster_state.get_circuit(
        qubits_matrix=qubits_matrix,
        theta=theta,
        phi=phi,
        probe_qubits=probe_qubits,
        basis=basis,
        anc_pairs=anc_pairs
    )

    assert isinstance(circuit, cirq.Circuit)
    assert len(circuit) > 0


def test_get_circuit_invalid_basis():
    """Test circuit creation with invalid basis states."""
    qubits_matrix = [
        [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)],
        [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]
    ]
    probe_qubits = [cirq.GridQubit(0, 0)]  # Only one probe qubit
    basis = [0, 0]  # Two basis states for one probe qubit

    with pytest.raises(ValueError):
        cluster_state.get_circuit(
            qubits_matrix=qubits_matrix,
            theta=np.pi/4,
            phi=np.pi/2,
            probe_qubits=probe_qubits,
            basis=basis,
            anc_pairs=[]
        ) 