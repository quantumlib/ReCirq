"""Implementation of cluster state circuits for measurement induced entanglement.

This module provides functionality for creating cluster states on Google's quantum
computer (Willow Pink) and implementing measurement protocols.
"""

from typing import List, Sequence, Tuple
import cirq
import cirq_google as cg
import numpy as np


def get_two_qubits_6x6(d: int = 6) -> Tuple[List[List[cirq.GridQubit]], List[cirq.GridQubit], List[List[List[cirq.GridQubit]]], np.ndarray]:
    """Creates a grid of qubits for cluster state experiments.
    
    Args:
        d: Size of the grid (3, 4, 5, or 6).
        
    Returns:
        A tuple containing:
            - qubits_matrix: 2D array of physical qubits
            - probe_qubits: List of probe qubits
            - anc_pairs: List of ancilla qubit pairs
            - all_qubits: Flattened array of all qubits
            
    Raises:
        ValueError: If d is not 3, 4, 5, or 6.
    """
    if d == 6:
        probe_qubits = [cirq.GridQubit(3, 4), cirq.GridQubit(3, 9)]
    elif d == 5:
        probe_qubits = [cirq.GridQubit(3, 4), cirq.GridQubit(3, 8)]
    elif d == 4:
        probe_qubits = [cirq.GridQubit(3, 5), cirq.GridQubit(3, 8)]
    elif d == 3:
        probe_qubits = [cirq.GridQubit(3, 5), cirq.GridQubit(3, 7)]
    else:
        raise ValueError("d must be 3, 4, 5, or 6")

    qubits_matrix = []
    for x in [3, 4, 5, 6, 7, 8]:
        qbs = []
        for y in [4, 5, 6, 7, 8, 9]:
            qbs.append(cirq.GridQubit(x, y))
        qubits_matrix.append(qbs)

    # Add ancilla pairs, [phy, anc]
    anc_pairs = [
        # Top row
        [[cirq.GridQubit(3, 4), cirq.GridQubit(2, 4)],
         [cirq.GridQubit(3, 5), cirq.GridQubit(2, 5)],
         [cirq.GridQubit(3, 6), cirq.GridQubit(2, 6)],
         [cirq.GridQubit(3, 7), cirq.GridQubit(2, 7)],
         [cirq.GridQubit(3, 8), cirq.GridQubit(2, 8)],
         [cirq.GridQubit(3, 9), cirq.GridQubit(2, 9)],
         # Bottom row
         [cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)],
         [cirq.GridQubit(8, 5), cirq.GridQubit(9, 5)],
         [cirq.GridQubit(8, 6), cirq.GridQubit(9, 6)],
         [cirq.GridQubit(8, 7), cirq.GridQubit(9, 7)],
         [cirq.GridQubit(8, 8), cirq.GridQubit(9, 8)],
         [cirq.GridQubit(8, 9), cirq.GridQubit(9, 9)],
         # Left column
         [cirq.GridQubit(3, 4), cirq.GridQubit(3, 3)],
         [cirq.GridQubit(4, 4), cirq.GridQubit(4, 3)],
         [cirq.GridQubit(5, 4), cirq.GridQubit(5, 3)],
         [cirq.GridQubit(6, 4), cirq.GridQubit(6, 3)],
         [cirq.GridQubit(7, 4), cirq.GridQubit(7, 3)],
         [cirq.GridQubit(8, 4), cirq.GridQubit(8, 3)],
         # Right column
         [cirq.GridQubit(3, 9), cirq.GridQubit(3, 10)],
         [cirq.GridQubit(4, 9), cirq.GridQubit(4, 10)],
         [cirq.GridQubit(5, 9), cirq.GridQubit(5, 10)],
         [cirq.GridQubit(6, 9), cirq.GridQubit(6, 10)],
         [cirq.GridQubit(7, 9), cirq.GridQubit(7, 10)],
         [cirq.GridQubit(8, 9), cirq.GridQubit(8, 10)]]
    ]

    # Get all qubits
    all_qubits = np.unique(np.concatenate(
        [np.array(qubits_matrix).flatten()] + 
        [np.array(anc_pair_set).flatten() for anc_pair_set in anc_pairs]))
    
    return qubits_matrix, probe_qubits, anc_pairs, all_qubits


def get_circuit(
    qubits_matrix: List[List[cirq.GridQubit]],
    theta: float,
    phi: float,
    probe_qubits: List[cirq.GridQubit],
    basis: List[int] = [0, 0],
    anc_pairs: List[List[List[cirq.GridQubit]]] = None
) -> cirq.Circuit:
    """Creates a quantum circuit for cluster state experiments.
    
    Args:
        qubits_matrix: 2D array of physical qubits.
        theta: Rotation angle for Y-axis.
        phi: Rotation angle for Z-axis.
        probe_qubits: List of probe qubits.
        basis: List of basis states for probe qubits (0=X, 1=Y, 2=Z).
        anc_pairs: List of ancilla qubit pairs.
        
    Returns:
        A Cirq circuit implementing the cluster state and measurements.
        
    Raises:
        ValueError: If the number of basis states doesn't match probe qubits.
    """
    if anc_pairs is None:
        anc_pairs = []
        
    circ = cirq.Circuit()
    qubits_list = np.array(qubits_matrix).flatten()
    num_qubits = len(qubits_list)
    num_col = len(qubits_matrix[0])
    num_row = len(qubits_matrix)
    
    if len(basis) != len(probe_qubits):
        raise ValueError("The number of basis states must match the number of probe qubits.")

    # Hadamard on all qubits
    circ.append([cirq.H(q) for q in qubits_list])

    # Horizontal bonds
    for i in range(num_row):
        for j in range(0, num_col - 1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i][j + 1]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i][j + 1]))
        for j in range(1, num_col - 1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i][j + 1]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i][j + 1]))

    # Vertical bonds
    for j in range(num_col):
        for i in range(0, num_row - 1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i + 1][j]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i + 1][j]))
        for i in range(1, num_row - 1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i + 1][j]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(
                axis_phase_exponent=0,
                x_exponent=0,
                z_exponent=-0.5).on(qubits_matrix[i + 1][j]))

    # Single qubit rotations
    for i in range(num_qubits):
        circ.append(cirq.rz(-phi).on(qubits_list[i]))
        circ.append(cirq.ry(-theta).on(qubits_list[i]))

    # Rotate probe qubits: x y z -> 0 1 2
    for q, b in zip(probe_qubits, basis):
        if b == 0:
            circ.append(cirq.H.on(q), strategy=cirq.circuits.InsertStrategy.INLINE)
        elif b == 1:
            circ.append(cirq.rx(np.pi / 2).on(q), strategy=cirq.circuits.InsertStrategy.INLINE)

    # Ancilla pairs
    for anc_pair_set in anc_pairs:
        for anc_pair in anc_pair_set:
            circ.append(cirq.H.on(anc_pair[1]), strategy=cirq.circuits.InsertStrategy.INLINE)
        for anc_pair in anc_pair_set:
            circ.append(cirq.CZ(*anc_pair))
        for anc_pair in anc_pair_set:
            circ.append(cirq.H.on(anc_pair[1]), strategy=cirq.circuits.InsertStrategy.INLINE)
    
    return circ 